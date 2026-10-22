import os
import random

import numpy as np
from PIL import Image
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, 'model', 'jjk_resnet50.onnx')
DST = os.path.join(BASE, 'model', 'jjk_resnet50_int8.onnx')
DS = os.path.join(BASE, 'dataset_v3', 'train')
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def img_to_tensor(path):
    img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - MEAN) / STD
    return a.transpose(2, 0, 1)[None]


files = [
    os.path.join(dp, f)
    for dp, _, fns in os.walk(DS)
    for f in fns
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
]
random.seed(0)
random.shuffle(files)
files = files[:300]
print(f'calibrating on {len(files)} images', flush=True)


class Reader(CalibrationDataReader):
    def __init__(self):
        self.it = iter(files)

    def get_next(self):
        try:
            return {'image': img_to_tensor(next(self.it))}
        except StopIteration:
            return None


quantize_static(
    SRC,
    DST,
    Reader(),
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,
)
print('quantized size MB:', os.path.getsize(DST) // 1048576, flush=True)

s = ort.InferenceSession(DST, providers=['CPUExecutionProvider'])
print('int8 session OK, inputs:', [(i.name, i.shape) for i in s.get_inputs()], flush=True)
