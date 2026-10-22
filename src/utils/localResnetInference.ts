import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const MODEL_DIR = path.resolve(__dirname, '../../model');
const ONNX_PATH = process.env.JJK_ONNX_PATH || path.join(MODEL_DIR, 'jjk_resnet50.onnx');
const CLASS_NAMES_PATH = path.join(MODEL_DIR, 'class_names.json');
const REPORT_PATH = path.join(MODEL_DIR, 'jjk_test_report.json');
const HISTORY_PATH = path.join(MODEL_DIR, 'training_history.json');
const SPLIT_COUNTS_PATH = path.join(MODEL_DIR, 'jjk_split_counts.json');

export interface LocalPrediction {
  id: string;
  probability: number;
}

export interface LocalInferenceResult {
  top: LocalPrediction[];
  latencyMs: number;
}

interface TestReport {
  overall: { top1: number; top5: number; test_images: number };
  per_class: Record<string, { correct: number; total: number; acc: number | null }>;
  confused: Array<[string, string, number]>;
}

let sessionPromise: Promise<import('onnxruntime-node').InferenceSession> | null = null;
let classNamesCache: string[] | null = null;
let reportCache: TestReport | null = null;
let historyCache: unknown = null;
let splitCountsCache: Record<string, { train: number; val: number; test: number }> | null = null;

export function localModelInstalled(): boolean {
  return fs.existsSync(ONNX_PATH) && fs.existsSync(CLASS_NAMES_PATH);
}

export function getClassNames(): string[] {
  if (!classNamesCache) {
    classNamesCache = JSON.parse(fs.readFileSync(CLASS_NAMES_PATH, 'utf-8')) as string[];
  }
  return classNamesCache ?? [];
}

export function getTestReport(): TestReport | null {
  if (!reportCache && fs.existsSync(REPORT_PATH)) {
    reportCache = JSON.parse(fs.readFileSync(REPORT_PATH, 'utf-8'));
  }
  return reportCache;
}

export function getTrainingHistory(): unknown {
  if (!historyCache && fs.existsSync(HISTORY_PATH)) {
    historyCache = JSON.parse(fs.readFileSync(HISTORY_PATH, 'utf-8'));
  }
  return historyCache;
}

export function getSplitCounts(): Record<string, { train: number; val: number; test: number }> | null {
  if (!splitCountsCache && fs.existsSync(SPLIT_COUNTS_PATH)) {
    splitCountsCache = JSON.parse(fs.readFileSync(SPLIT_COUNTS_PATH, 'utf-8'));
  }
  return splitCountsCache;
}

function getSession(): Promise<import('onnxruntime-node').InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const ort = await import('onnxruntime-node');
      return ort.InferenceSession.create(ONNX_PATH);
    })();
  }
  return sessionPromise;
}

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const IMG = 224;

export async function recognizeImageBase64(
  dataUriOrBase64: string,
  topK = 5
): Promise<LocalInferenceResult | null> {
  if (!localModelInstalled()) return null;

  const clean = dataUriOrBase64.replace(/^data:image\/[a-zA-Z0-9+.-]+;base64,/, '');
  const buffer = Buffer.from(clean, 'base64');
  if (buffer.length < 64) return null;

  const sharpMod = await import('sharp');
  const sharp = sharpMod.default;
  const base = sharp(buffer).flatten({ background: { r: 255, g: 255, b: 255 } });
  // Test-time augmentation: original + horizontal flip + zoomed center crop, averaged
  const views = await Promise.all([
    base.clone().resize(IMG, IMG, { fit: 'fill' }).raw().toBuffer({ resolveWithObject: true }),
    base.clone().flop().resize(IMG, IMG, { fit: 'fill' }).raw().toBuffer({ resolveWithObject: true }),
    base
      .clone()
      .resize(IMG + 32, IMG + 32, { fit: 'fill' })
      .extract({ left: 16, top: 16, width: IMG, height: IMG })
      .raw()
      .toBuffer({ resolveWithObject: true }),
  ]);

  const plane = IMG * IMG;
  const toPlane = (data: Buffer, offset: number, target: Float32Array) => {
    for (let i = 0; i < plane; i++) {
      const px = i * 3;
      target[offset + i] = (data[px] / 255 - MEAN[0]) / STD[0];
      target[offset + plane + i] = (data[px + 1] / 255 - MEAN[1]) / STD[1];
      target[offset + 2 * plane + i] = (data[px + 2] / 255 - MEAN[2]) / STD[2];
    }
  };
  const input = new Float32Array(views.length * 3 * plane);
  views.forEach((v, k) => toPlane(v.data, k * 3 * plane, input));

  const ort = await import('onnxruntime-node');
  const session = await getSession();
  const tensor = new ort.Tensor('float32', input, [views.length, 3, IMG, IMG]);
  const feeds: Record<string, import('onnxruntime-node').Tensor> = {
    [session.inputNames[0]]: tensor,
  };

  const t0 = Date.now();
  const outputs = await session.run(feeds);
  const logits = outputs[session.outputNames[0]].data as Float32Array;
  const latencyMs = Date.now() - t0;

  const nClasses = logits.length / views.length;
  const probs = new Array<number>(nClasses).fill(0);
  for (let k = 0; k < views.length; k++) {
    const off = k * nClasses;
    let max = -Infinity;
    for (let c = 0; c < nClasses; c++) max = Math.max(max, logits[off + c]);
    let sum = 0;
    const row = new Array<number>(nClasses);
    for (let c = 0; c < nClasses; c++) {
      row[c] = Math.exp(logits[off + c] - max);
      sum += row[c];
    }
    for (let c = 0; c < nClasses; c++) probs[c] += row[c] / sum / views.length;
  }
  const names = getClassNames();
  const ranked = probs
    .map((e, i) => ({ id: names[i], probability: e }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, topK)
    .map((p) => ({ id: p.id, probability: Number(p.probability.toFixed(4)) }));

  return { top: ranked, latencyMs };
}
