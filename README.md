# JJK ResNet-50 Vision Studio

A 64-class Jujutsu Kaisen character recognition studio built on a real, locally trained ResNet-50. Upload an image and the network runs in your browser (WebAssembly ONNX) or on the local server (fp32 ONNX) — no external AI APIs are involved anywhere.

**Live demo:** https://southeast-t693zdtll72.qoder.website

## What it does

- Recognizes all 64 MAPPA-era JJK wiki characters, including hard look-alikes (Noritoshi Kamo vs Suguru Geto vs Kenjaku, Maki vs Mai, Choso, Eso, Kechizu)
- Zero-click batch upload, drag & drop, or paste; per-character wiki frame galleries fetched live from the Jujutsu Kaisen Fandom wiki
- Real evaluation metrics shown in every tab — nothing is faked or hardcoded
- ResNet-50 architecture inspector, dataset registry with split counts, feature-map overlays, and the full training source

## Dataset

Raw wiki downloads: 13,460 files across the 64 character JSON manifests, filtered down to ~3,841 unique show frames (excluding manga panels, game art, fan art, and char-clone duplicates), split:

| Split | Train | Val | Test |
| --- | --- | --- | --- |
| Frames | 3,059 | 391 | 391 |

## Real model metrics

Held-out test evaluation on the 391-frame test split (3-view TTA, softmax):

| Model | Where it runs | Top-1 | Top-5 |
| --- | --- | --- | --- |
| ResNet-50 fp32 (93 MB) | Local Express server | 63.4% | 81.8% |
| ResNet-50 int8 QDQ (25 MB) | Browser (onnxruntime-web WASM) | 62.7% | 80.8% |

Fine-grained anime character identity is genuinely hard for a vision-only network: rejected experiments (v3 retrain, CLIP ViT-B/32 linear probe at 25.6%) and their artifacts are kept in the repo as evidence.

## Run locally

**Prerequisites:** Node.js 20+

```bash
npm install
npm run dev        # http://localhost:3000 — server-side fp32 inference
```

`npm run build` produces `dist/`, a fully static site that runs int8 inference entirely in the browser (open with any static file server). Append `?backend=browser` while running the dev server to preview the static-site behavior locally.

## Repository layout

```
server.ts                  Express API: model report, wiki image proxy, fp32 recognize endpoints
src/utils/
  localResnetInference.ts  Server-side ONNX inference (onnxruntime-node)
  browserInference.ts      Browser-side int8 ONNX inference (onnxruntime-web WASM)
  appBackend.ts            Server-first, static-fallback data facade
  wikiImages.ts            Browser-side Fandom wiki API client (CORS-friendly)
  jjkClassifierLogic.ts    64-class roster, calibrated class metadata, dataset manifest
src/components/            Workbench, dataset registry, architecture inspector, code viewer
public/model/              Shipped int8 ONNX + class names + eval reports + training history
train_v3.py                PyTorch training script (transfer learning, AdamW, cosine schedule)
quantize_int8.py           Static int8 QDQ quantization with calibration
eval_testset.ts            Test-split evaluator (writes report JSON via JJK_REPORT_OUT)
export_onnx.py             Checkpoint -> ONNX export with parity check
```

## Credits

Dataset images from the [Jujutsu Kaisen Fandom wiki](https://jujutsu-kaisen.fandom.com) (MAPPA anime keyframes and character art). Model, pipeline, and app built with Qoder.
