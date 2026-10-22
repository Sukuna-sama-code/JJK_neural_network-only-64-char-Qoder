import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip

BASE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(BASE, 'dataset_v3')
OUT_NPZ = os.path.join(BASE, 'model', 'clip_probe.npz')

device = 'cpu'
torch.set_num_threads(os.cpu_count() or 8)

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k'
)
model = model.to(device).eval()


def views(img):
    v1 = img.convert('RGB')
    v2 = v1.transpose(Image.FLIP_LEFT_RIGHT)
    v3 = v1.resize((256, 256), Image.BICUBIC).crop((16, 16, 240, 240))
    return [v1, v2, v3]


@torch.no_grad()
def feats_of(pil_images):
    batch = torch.stack([preprocess(im) for im in pil_images]).to(device)
    f = model.encode_image(batch)
    return F.normalize(f, dim=-1)


def load_split(name, tta):
    cache = os.path.join(BASE, 'model', f'clip_feats_{name}{"_tta" if tta else ""}.npz')
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        print(f'loaded cached feats for {name}', flush=True)
        return torch.from_numpy(z['X']), torch.from_numpy(z['Y']), list(z['classes'])
    root = os.path.join(DS, name)
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    X, Y = [], []
    for ci, cls in enumerate(classes):
        cdir = os.path.join(root, cls)
        for fn in sorted(os.listdir(cdir)):
            if not fn.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue
            img = Image.open(os.path.join(cdir, fn))
            vs = views(img) if tta else views(img)[:1]
            for v in vs:
                X.append(v)
                Y.append(ci)
    feats = []
    for i in range(0, len(X), 64):
        feats.append(feats_of(X[i:i + 64]))
    Xf = torch.cat(feats).cpu()
    Yt = torch.tensor(Y)
    np.savez(cache, X=Xf.numpy(), Y=Yt.numpy(), classes=np.array(classes))
    return Xf, Yt, classes


t0 = time.time()
Xtr, Ytr, CLS = load_split('train', tta=False)
print(f'train feats {tuple(Xtr.shape)} in {time.time() - t0:.0f}s', flush=True)
t0 = time.time()
Xva, Yva, _ = load_split('val', tta=True)
print(f'val feats {tuple(Xva.shape)} in {time.time() - t0:.0f}s', flush=True)
t0 = time.time()
Xte, Yte, _ = load_split('test', tta=True)
print(f'test feats {tuple(Xte.shape)} in {time.time() - t0:.0f}s', flush=True)

nC = len(CLS)

# --- classifier 1: cosine prototypes
protos = torch.stack([Xtr[Ytr == c].mean(dim=0) for c in range(nC)])
protos = F.normalize(protos, dim=-1)

# --- classifier 2: linear probe (LBFGS on train features)
W = torch.zeros(Xtr.shape[1], nC, requires_grad=True)
b = torch.zeros(nC, requires_grad=True)
opt = torch.optim.LBFGS([W, b], max_iter=150, tolerance_grad=1e-6)


def closure():
    opt.zero_grad()
    loss = F.cross_entropy(Xtr @ W + b, Ytr)
    loss.backward()
    return loss


t0 = time.time()
opt.step(closure)
print(f'linear probe trained in {time.time() - t0:.0f}s', flush=True)


def acc(scores, Y, k):
    pred = torch.topk(scores, k, dim=-1).indices
    return float((pred == Y[:, None]).any(dim=-1).float().mean())


def report(tag, X, Y, nviews):
    Yv = Y[::1]
    sp = X @ protos.T
    sl = X @ W + b
    lines = []
    for nm, s in (('proto', sp), ('linear', sl)):
        if nviews > 1:
            s = s.view(-1, nviews, s.shape[-1]).mean(dim=1)
            Yv = Y[::nviews]
        lines.append(f'{tag} {nm}: top-1 {acc(s, Yv, 1) * 100:.1f}% top-5 {acc(s, Yv, 5) * 100:.1f}%')
    print(' | '.join(lines), flush=True)
    return lines


report('VAL(tta3)', Xva, Yva, 3)
report('TEST(tta3)', Xte, Yte, 3)
report('TEST(single)', Xte[::3], Yte[::3], 1)

np.savez(
    OUT_NPZ,
    W=W.detach().numpy(),
    b=b.detach().numpy(),
    protos=protos.numpy(),
    classes=np.array(CLS),
)
print('saved', OUT_NPZ)
