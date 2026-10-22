import fs from 'fs';
import path from 'path';
import { JJK_MAPPA_ROSTER } from './src/utils/jjkClassifierLogic.ts';

const ROOT = path.resolve('dataset_v3');
const MAX_PER_CLASS = 200;
const SERVER = 'http://localhost:3000';

function mulberry32(a: number) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

async function pool<T>(items: T[], concurrency: number, fn: (item: T) => Promise<void>) {
  const queue = [...items];
  const workers = Array.from({ length: Math.min(concurrency, queue.length) }, async () => {
    while (queue.length) {
      const item = queue.shift()!;
      await fn(item).catch((e) => console.log('  skip:', String(e).slice(0, 80)));
    }
  });
  await Promise.all(workers);
}

interface Frame {
  wikiFileUrl: string;
  proxyUrl: string;
}

const totals: Record<string, number> = { train: 0, val: 0, test: 0 };

await pool(JJK_MAPPA_ROSTER, 4, async (char) => {
  const resp = await fetch(`${SERVER}/api/character-images/${char.id}`);
  const data = (await resp.json()) as { images: Frame[] };
  const seen = new Set<string>();
  const frames = data.images
    .filter((f) => {
      const u = f.wikiFileUrl;
      if (!u || seen.has(u)) return false;
      seen.add(u);
      return true;
    })
    .slice(0, MAX_PER_CLASS);

  const rnd = mulberry32(char.classIndex * 31 + 7);
  const order = frames.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  const n = frames.length;
  const nTest = Math.max(1, Math.round(n * 0.1));
  const nVal = Math.max(1, Math.round(n * 0.1));
  const splitAt = (p: number) => (p < nTest ? 'test' : p < nTest + nVal ? 'val' : 'train');

  const tasks = order.map((frameIdx, pos) => ({ frame: frames[frameIdx], split: splitAt(pos), pos }));
  let saved = 0;
  await pool(tasks, 8, async ({ frame, split, pos }) => {
    const ext = path.extname(new URL(frame.wikiFileUrl).pathname) || '.jpg';
    const dir = path.join(ROOT, split, char.id);
    fs.mkdirSync(dir, { recursive: true });
    const file = path.join(dir, `${String(pos).padStart(3, '0')}${ext}`);
    if (fs.existsSync(file) && fs.statSync(file).size > 2048) {
      saved++;
      return;
    }
    const r = await fetch(SERVER + frame.proxyUrl);
    if (!r.ok) return;
    const buf = Buffer.from(await r.arrayBuffer());
    if (buf.length < 2048) return;
    fs.writeFileSync(file, buf);
    totals[split]++;
    saved++;
  });
  console.log(`${char.id.padEnd(22)} ${String(saved).padStart(3)} frames`);
});

fs.mkdirSync(ROOT, { recursive: true });
fs.writeFileSync(path.join(ROOT, 'manifest.json'), JSON.stringify(totals, null, 2));
console.log('DATASET v3 TOTALS:', totals);
