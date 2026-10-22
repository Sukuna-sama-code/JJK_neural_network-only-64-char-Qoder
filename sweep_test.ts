import { JJK_MAPPA_ROSTER } from './src/utils/jjkClassifierLogic.ts';
import { recognizeImageBase64 } from './src/utils/localResnetInference.ts';

let top1 = 0;
let top5 = 0;
let done = 0;
const misses: string[] = [];

for (const char of JJK_MAPPA_ROSTER) {
  const url = char.canonicalPortraitUrl.split('/revision/')[0];
  try {
    const resp = await fetch(url, {
      headers: { 'User-Agent': 'Mozilla/5.0', Referer: 'https://jujutsu-kaisen.fandom.com/' },
    });
    if (!resp.ok) {
      misses.push(`${char.id}: http ${resp.status}`);
      continue;
    }
    const b64 = Buffer.from(await resp.arrayBuffer()).toString('base64');
    const res = await recognizeImageBase64(b64, 5);
    if (!res) {
      misses.push(`${char.id}: no result`);
      continue;
    }
    const ids = res.top.map((p) => p.id);
    const hit1 = ids[0] === char.id;
    const hit5 = ids.includes(char.id);
    if (hit1) top1++;
    if (hit5) top5++;
    done++;
    if (!hit1) misses.push(`${char.id} -> guessed ${ids[0]} (${(res.top[0].probability * 100).toFixed(1)}%)`);
    console.log(
      `${hit1 ? 'OK  ' : 'MISS'} ${char.id.padEnd(22)} top1=${ids[0].padEnd(22)} ${(
        res.top[0].probability * 100
      ).toFixed(1)}%`
    );
  } catch (e) {
    misses.push(`${char.id}: ${String(e).slice(0, 60)}`);
  }
}

console.log(
  `\nCANONICAL PORTRAIT SWEEP: top-1 ${top1}/${done} (${((top1 / Math.max(1, done)) * 100).toFixed(
    1
  )}%) | top-5 ${top5}/${done} (${((top5 / Math.max(1, done)) * 100).toFixed(1)}%)`
);
if (misses.length) console.log('misses/errors:\n' + misses.join('\n'));
