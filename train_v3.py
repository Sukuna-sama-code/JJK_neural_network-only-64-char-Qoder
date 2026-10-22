import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'dataset_v3')
DEVICE = 'cpu'
EPOCHS = 25
BATCH = 64

torch.set_num_threads(os.cpu_count() or 4)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ToTensor(),
    normalize,
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])

train_ds = ImageFolder(os.path.join(DATA, 'train'), transform=train_tf)
val_ds = ImageFolder(os.path.join(DATA, 'val'), transform=val_tf)
test_ds = ImageFolder(os.path.join(DATA, 'test'), transform=val_tf)

# Align val/test labels to the train folder's class indexing (ImageFolder renames per-folder)
for ds in (val_ds, test_ds):
    remap = {old: train_ds.class_to_idx[name] for name, old in ds.class_to_idx.items()}
    ds.targets = [remap[t] for t in ds.targets]
    ds.samples = [(p, remap[t]) for p, t in ds.samples]

CLASSES = train_ds.classes
print('classes:', len(CLASSES), '| train:', len(train_ds), '| val:', len(val_ds), '| test:', len(test_ds))

counts = torch.bincount(torch.tensor(train_ds.targets), minlength=len(CLASSES)).float()
weights = 1.0 / counts[torch.tensor(train_ds.targets)]
sampler = WeightedRandomSampler(weights, num_samples=len(train_ds) * 2, replacement=True)

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Dropout(0.35),
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(512, len(CLASSES)),
)
for name in ('conv1', 'bn1', 'layer1'):
    for p in getattr(model, name).parameters():
        p.requires_grad = False

backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('fc')]
head_params = [p for n, p in model.named_parameters() if n.startswith('fc')]
opt = torch.optim.AdamW([
    {'params': backbone_params, 'lr': 5e-5},
    {'params': head_params, 'lr': 3e-4},
], weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

train_dl = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)


def evaluate(dl):
    model.eval()
    correct = top5 = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in dl:
            out = model(x)
            loss_sum += loss_fn(out, y).item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            top5 += (out.topk(5, 1).indices == y[:, None]).any(1).sum().item()
            total += len(y)
    return loss_sum / max(1, total), correct / max(1, total), top5 / max(1, total)


history = []
best_val = 0.0
best_epoch = 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    running = 0.0
    correct = 0
    total = 0
    for x, y in train_dl:
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        running += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    sched.step()
    val_loss, val_acc, val_top5 = evaluate(val_dl)
    train_acc = correct / max(1, total)
    print(
        f'ep {epoch:02d} | {time.time() - t0:6.1f}s | train loss {running / max(1, total):.3f} acc {train_acc:.3f} '
        f'| val loss {val_loss:.3f} acc {val_acc:.3f} top5 {val_top5:.3f}',
        flush=True,
    )
    history.append({
        'epoch': epoch,
        'trainLoss': round(running / max(1, total), 4),
        'trainAcc': round(train_acc, 4),
        'valLoss': round(val_loss, 4),
        'valAcc': round(val_acc, 4),
        'valTop5': round(val_top5, 4),
    })
    if val_acc > best_val:
        best_val = val_acc
        best_epoch = epoch
        torch.save({'model': model.state_dict(), 'classes': CLASSES}, os.path.join(BASE, 'model', 'best_jjk_resnet50_v3.pt'))
        print('  saved best checkpoint', flush=True)

print(f'BEST val top-1 {best_val:.3f} @ epoch {best_epoch}')

model.load_state_dict(torch.load(os.path.join(BASE, 'model', 'best_jjk_resnet50_v3.pt'), map_location='cpu')['model'])
test_loss, test_acc, test_top5 = evaluate(test_dl)
print(f'TEST top-1 {test_acc:.3f} | top-5 {test_top5:.3f} | images {len(test_ds)}')

model.eval()
per_class = {}
confused = {}
with torch.no_grad():
    for x, y in test_dl:
        pred = model(x).argmax(1)
        for t, p in zip(y.tolist(), pred.tolist()):
            cid = CLASSES[t]
            per_class.setdefault(cid, {'correct': 0, 'total': 0})
            per_class[cid]['total'] += 1
            if t == p:
                per_class[cid]['correct'] += 1
            else:
                key = (cid, CLASSES[p])
                confused[key] = confused.get(key, 0) + 1
for cid, v in per_class.items():
    v['acc'] = round(v['correct'] / v['total'], 4) if v['total'] else None
report = {
    'overall': {'top1': round(test_acc, 4), 'top5': round(test_top5, 4), 'test_images': len(test_ds)},
    'per_class': per_class,
    'confused': [[a, b, c] for (a, b), c in sorted(confused.items(), key=lambda kv: -kv[1])[:40]],
}
with open(os.path.join(BASE, 'model', 'jjk_test_report_v3.json'), 'w') as f:
    json.dump(report, f, indent=1)
with open(os.path.join(BASE, 'model', 'training_history_v3.json'), 'w') as f:
    json.dump({
        'setup': {
            'epochs': EPOCHS, 'batchSize': BATCH,
            'optimizer': 'AdamW (backbone lr 5e-5, head lr 3e-4, wd 0.01)',
            'scheduler': 'CosineAnnealingLR', 'loss': 'CE label_smoothing=0.05',
            'frozenStages': 'conv1,bn1,layer1',
            'trainImages': len(train_ds), 'valImages': len(val_ds), 'testImages': len(test_ds),
            'numClasses': len(CLASSES),
        },
        'best': {'epoch': best_epoch, 'valTop1': round(best_val, 4)},
        'test': {'top1': round(test_acc, 4), 'top5': round(test_top5, 4), 'images': len(test_ds)},
        'epochs_history': history,
    }, f, indent=1)
print('reports saved')
