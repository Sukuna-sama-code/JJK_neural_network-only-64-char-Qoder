import fs from 'fs';
import path from 'path';
import { recognizeImageBase64 } from './src/utils/localResnetInference.ts';

const TEST_DIR = path.resolve('dataset_v3/test');
const classes = fs.readdirSync(TEST_DIR).filter((d) => fs.statSync(path.join(TEST_DIR, d)).isDirectory());

let top1 = 0;
let top5 = 0;
let done = 0;
const misses: string[] = [];
const perClass: Record<string, { correct: number; total: number }> = {};
const confusion = new Map<string, number>();

for (const cls of classes) {
  perClass[cls] = { correct: 0, total: 0 };
  const dir = path.join(TEST_DIR, cls);
  const files = fs.readdirSync(dir).filter((f) => /\.(jpe?g|png|webp)$/i.test(f));
  for (const f of files) {
    perClass[cls].total++;
    const b64 = fs.readFileSync(path.join(dir, f)).toString('base64');
    const res = await recognizeImageBase64(b64, 5);
    if (!res) continue;
    const ids = res.top.map((p) => p.id);
    if (ids[0] === cls) {
      top1++;
      perClass[cls].correct++;
    }
    if (ids.includes(cls)) top5++;
    else misses.push(`${cls}/${f} -> ${ids[0]}`);
    if (ids[0] !== cls) {
      const key = `${cls}->${ids[0]}`;
      confusion.set(key, (confusion.get(key) || 0) + 1);
    }
    done++;
  }
}

const reportOut = process.env.JJK_REPORT_OUT;
if (reportOut) {
  const confused = [...confusion.entries()]
    .map(([k, n]) => k.split('->').concat(String(n)) as [string, string, string])
    .sort((a, b) => Number(b[2]) - Number(a[2]))
    .slice(0, 20)
    .map(([from, to, n]) => [from, to, Number(n)] as [string, string, number]);
  const per_class: Record<string, { correct: number; total: number; acc: number | null }> = {};
  for (const [cls, v] of Object.entries(perClass)) {
    per_class[cls] = { ...v, acc: v.total > 0 ? v.correct / v.total : null };
  }
  const report = {
    overall: {
      top1: Number((top1 / Math.max(1, done)).toFixed(4)),
      top5: Number((top5 / Math.max(1, done)).toFixed(4)),
      test_images: done,
    },
    per_class,
    confused,
  };
  fs.writeFileSync(reportOut, JSON.stringify(report, null, 2));
  console.log(`report written: ${reportOut}`);
}

console.log(
  `TESTSET EVAL (${path.basename(process.env.JJK_ONNX_PATH || 'jjk_resnet50.onnx')}): top-1 ${top1}/${done} (${(
    (top1 / Math.max(1, done)) * 100
  ).toFixed(1)}%) | top-5 ${top5}/${done} (${((top5 / Math.max(1, done)) * 100).toFixed(1)}%)`
);
console.log('miss count:', misses.length);
