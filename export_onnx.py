import json
import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

BASE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(BASE, 'model', 'best_jjk_resnet50.pt')
OUT = os.path.join(BASE, 'model', 'jjk_resnet50.onnx')

ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
print('checkpoint keys:', list(ckpt.keys()))

model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.35),
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(512, 64),
)
model.load_state_dict(ckpt['model'])
model.eval()

classes = list(ckpt['classes'])
with open(os.path.join(BASE, 'model', 'class_names.json')) as f:
    names = json.load(f)
assert classes == list(names), f'class order mismatch: {classes[:3]} vs {names[:3]}'
print('classes verified:', len(classes))

dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy,
    OUT,
    input_names=['image'],
    output_names=['logits'],
    dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}},
    opset_version=17,
    dynamo=False,
)
print('exported:', os.path.getsize(OUT) // (1024 * 1024), 'MB')

import onnxruntime as ort

sess = ort.InferenceSession(OUT, providers=['CPUExecutionProvider'])
with torch.no_grad():
    ref = model(dummy).numpy()
got = sess.run(None, {'image': dummy.numpy()})[0]
diff = float(np.abs(ref - got).max())
print('parity check max abs diff:', diff)
assert diff < 1e-3, 'ONNX output does not match PyTorch model'
print('OK: single-file ONNX matches the trained model')
