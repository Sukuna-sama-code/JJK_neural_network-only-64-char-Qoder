import { TestReportShape } from './recognitionResult';

// Browser-side ONNX ResNet-50: same preprocessing as the local server path
// (white-flatten, squish to 224, ImageNet normalize, 3-view TTA softmax average).

const MODEL_URL = '/model/jjk_resnet50_int8.onnx';
const ORT_WASM_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
const IMG = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let sessionPromise: Promise<import('onnxruntime-web').InferenceSession> | null = null;
let classNamesCache: string[] | null = null;
let staticReportCache: TestReportShape | null = null;
let staticReportTried = false;

async function getSession(): Promise<import('onnxruntime-web').InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const ort = await import('onnxruntime-web');
      ort.env.wasm.wasmPaths = ORT_WASM_CDN;
      ort.env.wasm.numThreads = 1;
      return ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
    })();
  }
  return sessionPromise;
}

async function fetchJsonSafe<T>(url: string): Promise<T | null> {
  try {
    const r = await fetch(url);
    if (!r.ok) return null;
    return (await r.json()) as T;
  } catch {
    return null;
  }
}

export async function getBrowserClassNames(): Promise<string[]> {
  if (classNamesCache && classNamesCache.length > 0) return classNamesCache;
  const names = await fetchJsonSafe<string[]>('/model/class_names.json');
  classNamesCache = Array.isArray(names) ? names : [];
  return classNamesCache;
}

// Cache-buster: these JSON files are content-updated between releases without
// hashed names, so repeat visitors could otherwise be pinned to a stale
// cached copy by an edge node answering 304 against old validators.
const STATIC_ASSET_V = '?v=20261022';

export async function getStaticReport(): Promise<TestReportShape | null> {
  if (staticReportTried) return staticReportCache;
  staticReportTried = true;
  staticReportCache = await fetchJsonSafe<TestReportShape>(
    `/model/jjk_test_report.json${STATIC_ASSET_V}`
  );
  return staticReportCache;
}

function drawView(
  source: CanvasImageSource,
  natW: number,
  natH: number,
  kind: 'orig' | 'flop' | 'zoom'
): Uint8ClampedArray {
  const canvas = document.createElement('canvas');
  canvas.width = IMG;
  canvas.height = IMG;
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, IMG, IMG);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  if (kind === 'orig') {
    ctx.drawImage(source, 0, 0, natW, natH, 0, 0, IMG, IMG);
  } else if (kind === 'flop') {
    ctx.save();
    ctx.translate(IMG, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(source, 0, 0, natW, natH, 0, 0, IMG, IMG);
    ctx.restore();
  } else {
    const mid = document.createElement('canvas');
    mid.width = IMG + 32;
    mid.height = IMG + 32;
    const mCtx = mid.getContext('2d')!;
    mCtx.fillStyle = '#ffffff';
    mCtx.fillRect(0, 0, IMG + 32, IMG + 32);
    mCtx.drawImage(source, 0, 0, natW, natH, 0, 0, IMG + 32, IMG + 32);
    ctx.drawImage(mid, 16, 16, IMG, IMG, 0, 0, IMG, IMG);
  }
  return ctx.getImageData(0, 0, IMG, IMG).data;
}

export interface BrowserTopPrediction {
  id: string;
  probability: number;
}

export async function browserRecognize(
  imageSource: string,
  topK = 5
): Promise<{ top: BrowserTopPrediction[]; latencyMs: number } | null> {
  const names = await getBrowserClassNames();
  if (names.length === 0) return null;

  const img = new Image();
  img.crossOrigin = 'anonymous';
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error('image decode failed'));
    img.src = imageSource;
  });

  const natW = img.naturalWidth || img.width;
  const natH = img.naturalHeight || img.height;
  if (!natW || !natH) return null;

  const plane = IMG * IMG;
  const views: Uint8ClampedArray[] = [
    drawView(img, natW, natH, 'orig'),
    drawView(img, natW, natH, 'flop'),
    drawView(img, natW, natH, 'zoom'),
  ];
  const input = new Float32Array(views.length * 3 * plane);
  views.forEach((px, k) => {
    const offset = k * 3 * plane;
    for (let i = 0; i < plane; i++) {
      const p = i * 3;
      input[offset + i] = (px[p] / 255 - MEAN[0]) / STD[0];
      input[offset + plane + i] = (px[p + 1] / 255 - MEAN[1]) / STD[1];
      input[offset + 2 * plane + i] = (px[p + 2] / 255 - MEAN[2]) / STD[2];
    }
  });

  const ort = await import('onnxruntime-web');
  const session = await getSession();
  const tensor = new ort.Tensor('float32', input, [views.length, 3, IMG, IMG]);
  const feeds: Record<string, import('onnxruntime-web').Tensor> = {
    [session.inputNames[0]]: tensor,
  };

  const t0 = performance.now();
  const outputs = await session.run(feeds);
  const latencyMs = Number((performance.now() - t0).toFixed(1));
  const logits = outputs[session.outputNames[0]].data as Float32Array;

  const nClasses = logits.length / views.length;
  const probs = new Array<number>(nClasses).fill(0);
  for (let k = 0; k < views.length; k++) {
    const row = logits.subarray(k * nClasses, (k + 1) * nClasses);
    let max = -Infinity;
    for (let i = 0; i < nClasses; i++) max = Math.max(max, row[i]);
    let sum = 0;
    const exps = new Array<number>(nClasses);
    for (let i = 0; i < nClasses; i++) {
      exps[i] = Math.exp(row[i] - max);
      sum += exps[i];
    }
    for (let i = 0; i < nClasses; i++) probs[i] += exps[i] / sum / views.length;
  }

  const ranked = probs
    .map((e, i) => ({ id: names[i], probability: e }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, topK)
    .map((p) => ({ id: p.id, probability: Number(p.probability.toFixed(4)) }));

  return { top: ranked, latencyMs };
}
