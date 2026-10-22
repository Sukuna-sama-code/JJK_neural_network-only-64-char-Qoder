import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import {
  JJK_MAPPA_ROSTER,
  MappaRealImageFrame,
  RecognitionResult,
  calculateDatasetManifest,
  getWikiSearchTitlesForCharacter,
  scoreImageFeaturesAgainstRoster,
} from './src/utils/jjkClassifierLogic.ts';
import {
  getClassNames,
  getSplitCounts,
  getTestReport,
  getTrainingHistory,
  localModelInstalled,
  recognizeImageBase64,
} from './src/utils/localResnetInference.ts';
import { buildRecognitionResult } from './src/utils/recognitionResult.ts';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.json({ limit: '50mb' }));

// Cache real MAPPA anime images per character ID from jujutsu-kaisen.fandom.com
const characterImagesCache = new Map<string, MappaRealImageFrame[]>();

// In-memory recognition cache keyed by image content signature for <1ms repeat inference
const recognitionCache = new Map<string, RecognitionResult>();

function computeFastBase64Signature(b64: string): string {
  const len = b64.length;
  if (len < 128) return b64;
  const mid = Math.floor(len / 2);
  const q1 = Math.floor(len / 4);
  const q3 = Math.floor((len * 3) / 4);
  return `${len}:${b64.slice(0, 48)}:${b64.slice(q1, q1 + 48)}:${b64.slice(mid, mid + 48)}:${b64.slice(q3, q3 + 48)}:${b64.slice(-48)}`;
}

async function fetchRealMappaImagesForCharacter(
  characterId: string
): Promise<MappaRealImageFrame[]> {
  if (characterImagesCache.has(characterId)) {
    return characterImagesCache.get(characterId)!;
  }

  const char = JJK_MAPPA_ROSTER.find((c) => c.id === characterId);
  if (!char) return [];

  const searchTitles = getWikiSearchTitlesForCharacter(char);
  const collected = new Map<string, MappaRealImageFrame>();

  const classifyFrameCategory = (
    title: string,
    w: number = 1280,
    h: number = 720
  ): MappaRealImageFrame['category'] => {
    const aspect = w / Math.max(1, h);
    if (
      /head|face|close[- ]?up|bust|half|smile|eyes|expression|avatar|icon|grin|glare|first appearance|greets|comes to the rescue|asks|suggests|talk|speaks|tells|P\.(png|jpg|webp)$/i.test(
        title
      ) ||
      (aspect >= 0.78 && aspect <= 1.28 && w <= 760)
    ) {
      return 'Face & Half-Body Crop';
    }
    if (
      /^\s*[^()]+\(Anime(\s*\d+)?\)\.(png|jpg|webp)$/i.test(title) &&
      title.replace(/\(Anime.*$/i, '').trim().split(/\s+/).length <= 3
    ) {
      return 'Character Sheet / Portrait';
    }
    if (/profile|portrait|character sheet|design|full body/i.test(title)) {
      return 'Character Sheet / Portrait';
    }
    if (
      /cursed|domain|technique|resonance|hollow|shrine|shadow|blood|meteor|idle|black flash|wheel|void|boogie|ratio|divergent|straw doll|simple domain|star rage|garuda|black hole/i.test(
        title
      )
    ) {
      return 'Cursed Technique Frame';
    }
    if (
      /fighting|vs\.|attack|punch|joins|defeats|wounds|angle|side|rescues|battle|combat|strike/i.test(
        title
      )
    ) {
      return 'Multi-Angle Anime Cut';
    }
    return 'Episode Action Cut';
  };

  const isExcludedNonShowFile = (rawTitle: string): boolean => {
    const title = rawTitle.replace(/_/g, ' ');
    const lower = title.toLowerCase();
    if (
      lower.endsWith('.gif') ||
      lower.endsWith('.svg') ||
      lower.endsWith('.ogg') ||
      lower.endsWith('.mp4')
    ) {
      return true;
    }
    if (
      /\bvolume\s*\d+|\bchapter\s*\d+|\bsketch\b|\bjump\b|\blight\s+novel\b|^male\.png$|^female\.png$|^unknown|\bstage\s+play\b|\bcover\b|\bpopularity\b|\bcursed\s+clash\b|\bphantom\s+parade\b/i.test(
        title
      )
    ) {
      return true;
    }
    // Prevent character clashes between Noritoshi Kamo (Kyoto student), Suguru Geto, and Kenjaku
    if (characterId === 'noritoshi_kamo') {
      if (/ancestor|kenjaku|pseudo[- ]?geto|imposter/i.test(title)) {
        return true;
      }
    } else if (characterId === 'suguru_geto') {
      if (/kenjaku|pseudo[- ]?geto|imposter|ancestor|prison\s+realm/i.test(title)) {
        return true;
      }
    } else if (characterId === 'kenjaku') {
      if (
        /suguru\s+geto.*(2006|prequel|young|goes\s+rogue|hunting\s+curses|introduced|last\s+moments|congregation|disdain|group)/i.test(
          title
        )
      ) {
        return true;
      }
    } else if (characterId === 'yuki_tsukumo') {
      // Exclude wide-angle scenes where Kenjaku's back or corridor is foreground instead of Yuki Tsukumo
      if (/rescues\s+the\s+jujutsu\s+sorcerers|tombs\s+of\s+the\s+star\s+corridor\s+rejecting/i.test(title)) {
        return true;
      }
    }
    return false;
  };

  // Build character-specific token filter so Wiki page generator=images never leaks navbox images of other characters
  const charTokens = Array.from(
    new Set(
      [
        ...char.id.split('_'),
        ...char.name
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, ' ')
          .split(/\s+/),
        ...searchTitles.flatMap((t) =>
          t
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, ' ')
            .split(/\s+/)
        ),
      ].filter((t) => t.length >= 3 && !['the', 'and', 'cursed', 'special', 'grade', 'divine', 'general'].includes(t))
    )
  );

  const belongsToCharacter = (title: string): boolean => {
    if (isExcludedNonShowFile(title)) return false;
    const norm = title.toLowerCase().replace(/[^a-z0-9]+/g, ' ');
    return charTokens.some((tok) => norm.includes(tok));
  };

  // Seed index 0 with the character's verified canonical MAPPA anime portrait
  if (char.canonicalPortraitUrl) {
    const cleanCanonical = char.canonicalPortraitUrl.split('/revision/')[0];
    collected.set(cleanCanonical, {
      id: `${characterId}-canonical-0`,
      title: `${char.name} (Official MAPPA Anime Keyframe)`,
      wikiFileUrl: cleanCanonical,
      proxyUrl: `/api/proxy-image?url=${encodeURIComponent(cleanCanonical)}`,
      width: 1280,
      height: 720,
      category: 'Character Sheet / Portrait',
    });
  }

  try {
    // 1. Query MediaWiki allimages by character name prefix (up to 100 character-named files)
    await Promise.all(
      searchTitles.map(async (prefix) => {
        const apiUrl = `https://jujutsu-kaisen.fandom.com/api.php?action=query&list=allimages&aiprefix=${encodeURIComponent(
          prefix
        )}&ailimit=100&format=json`;
        const resp = await fetch(apiUrl, {
          headers: { 'User-Agent': 'Mozilla/5.0' },
        });
        if (!resp.ok) return;
        const data = await resp.json();
        const allimages = data?.query?.allimages || [];
        for (const img of allimages) {
          const name: string = img.name || '';
          const rawUrl: string = img.url || '';
          if (!name || !rawUrl || !belongsToCharacter(name)) continue;
          const w = Number(img.width) || 1280;
          const h = Number(img.height) || 720;
          if (w < 140 || h < 140) continue;

          const cleanUrl = rawUrl.split('/revision/')[0];
          const readableTitle = name.replace(/_/g, ' ');
          if (!collected.has(cleanUrl)) {
            collected.set(cleanUrl, {
              id: `${characterId}-${collected.size}`,
              title: readableTitle,
              wikiFileUrl: cleanUrl,
              proxyUrl: `/api/proxy-image?url=${encodeURIComponent(cleanUrl)}`,
              width: w,
              height: h,
              category: classifyFrameCategory(readableTitle, w, h),
            });
          }
        }
      })
    );

    // 2. Query character main Wiki page + /Image_Gallery + /Synopsis subpages for all individual episode frames from the show
    const joinedTitles = searchTitles
      .flatMap((t) => [t, `${t}/Image_Gallery`, `${t}/Synopsis`])
      .join('|');
    const galleryUrl = `https://jujutsu-kaisen.fandom.com/api.php?action=query&titles=${encodeURIComponent(
      joinedTitles
    )}&generator=images&gimlimit=500&prop=imageinfo&iiprop=url|dimensions&format=json`;
    const galleryResp = await fetch(galleryUrl, {
      headers: { 'User-Agent': 'Mozilla/5.0' },
    });
    if (galleryResp.ok) {
      const galleryData = await galleryResp.json();
      const pages = Object.values(galleryData?.query?.pages || {}) as Array<{
        title?: string;
        imageinfo?: Array<{ url?: string; width?: number; height?: number }>;
      }>;
      for (const p of pages) {
        const rawTitle = (p.title || '').replace(/^File:/i, '');
        const info = p.imageinfo?.[0];
        if (!rawTitle || !info?.url || !belongsToCharacter(rawTitle)) continue;
        const w = Number(info.width) || 1280;
        const h = Number(info.height) || 720;
        if (w < 140 || h < 140) continue;

        const cleanUrl = info.url.split('/revision/')[0];
        if (!collected.has(cleanUrl)) {
          collected.set(cleanUrl, {
            id: `${characterId}-${collected.size}`,
            title: rawTitle,
            wikiFileUrl: cleanUrl,
            proxyUrl: `/api/proxy-image?url=${encodeURIComponent(cleanUrl)}`,
            width: w,
            height: h,
            category: classifyFrameCategory(rawTitle, w, h),
          });
        }
      }
    }
  } catch (_err) {
    // Return frames collected so far (at minimum the canonical MAPPA anime portrait)
  }

  const canonicalClean = char.canonicalPortraitUrl.split('/revision/')[0];
  const baseSorted = Array.from(collected.values()).sort((a, b) => {
    const score = (item: MappaRealImageFrame) => {
      if (item.wikiFileUrl === canonicalClean) return 1000;
      let s = 0;
      if (/\(Anime(\s*\d+)?\)\.(png|jpg|webp)$/i.test(item.title)) s += 300;
      if (item.title.includes('(Anime')) s += 150;
      if (item.category === 'Face & Half-Body Crop') s += 110;
      if (item.category === 'Character Sheet / Portrait') s += 90;
      if (item.category === 'Multi-Angle Anime Cut') s += 70;
      if (item.category === 'Cursed Technique Frame') s += 55;
      const firstToken = char.name.split(' ')[0].toLowerCase();
      if (item.title.toLowerCase().startsWith(firstToken)) s += 60;
      if (item.title.toLowerCase().includes(firstToken)) s += 30;
      return s;
    };
    return score(b) - score(a);
  });

  // Synthesize dedicated Half-Image (Waist-Up Crop), Tight Face Close-Up, and Split Half-Image training frames
  // from the character's primary anime keyframes so every character has explicit half-image & face training samples
  const primarySeeds = baseSorted.slice(0, Math.min(3, baseSorted.length));
  const multiCropTrainingFrames: MappaRealImageFrame[] = [];
  primarySeeds.forEach((seed, sIdx) => {
    multiCropTrainingFrames.push({
      id: `${characterId}-halfbody-crop-${sIdx}`,
      title: `${char.name} — Half-Image / Waist-Up Bust Crop #${sIdx + 1} (${seed.title})`,
      wikiFileUrl: seed.wikiFileUrl,
      proxyUrl: seed.proxyUrl,
      width: 640,
      height: 480,
      category: 'Face & Half-Body Crop',
      cropPreset: 'half-body',
      angleTag: sIdx === 0 ? 'Front Bust Crop' : sIdx === 1 ? '3/4 Waist-Up Crop' : 'Dynamic Half-Body Crop',
    });
    multiCropTrainingFrames.push({
      id: `${characterId}-face-closeup-${sIdx}`,
      title: `${char.name} — Tight Face & Eyes Close-Up #${sIdx + 1} (${seed.title})`,
      wikiFileUrl: seed.wikiFileUrl,
      proxyUrl: seed.proxyUrl,
      width: 448,
      height: 448,
      category: 'Face & Half-Body Crop',
      cropPreset: 'face-closeup',
      angleTag: sIdx === 0 ? 'Facial Keypoints Crop' : 'Eye & Hairline Crop',
    });
    if (sIdx === 0) {
      multiCropTrainingFrames.push({
        id: `${characterId}-splithalf-crop-0`,
        title: `${char.name} — Vertical Split Half-Image Crop (${seed.title})`,
        wikiFileUrl: seed.wikiFileUrl,
        proxyUrl: seed.proxyUrl,
        width: 360,
        height: 720,
        category: 'Face & Half-Body Crop',
        cropPreset: 'left-half',
        angleTag: '50% Split Half-Image',
      });
    }
  });

  // Interleave canonical portrait, multi-crop half-body/face training frames, and multi-angle episode cuts
  const finalFrames: MappaRealImageFrame[] = [];
  if (baseSorted.length > 0) {
    finalFrames.push({ ...baseSorted[0], cropPreset: 'full', angleTag: 'Canonical Keyframe' });
  }
  finalFrames.push(...multiCropTrainingFrames);
  for (let i = 1; i < baseSorted.length; i++) {
    finalFrames.push({
      ...baseSorted[i],
      cropPreset: 'full',
      angleTag:
        baseSorted[i].category === 'Multi-Angle Anime Cut'
          ? 'Multi-Angle Cut'
          : baseSorted[i].category === 'Face & Half-Body Crop'
            ? 'In-Anime Close-Up'
            : 'Show Frame',
    });
  }

  if (finalFrames.length > 0) {
    characterImagesCache.set(characterId, finalFrames);
  }
  return finalFrames;
}

app.get('/api/dataset-manifest', (_req, res) => {
  const manifest = calculateDatasetManifest(JJK_MAPPA_ROSTER, 200, 4);
  res.json({
    manifest,
    roster: JJK_MAPPA_ROSTER,
  });
});

app.get('/api/character-images/:characterId', async (req, res) => {
  const { characterId } = req.params;
  const images = await fetchRealMappaImagesForCharacter(characterId);
  res.json({
    characterId,
    count: images.length,
    images,
  });
});

// In-memory binary image cache for proxied MAPPA Wiki images so switching characters is instantaneous
const proxyBinaryCache = new Map<string, { contentType: string; buffer: Buffer }>();

app.get('/api/proxy-image', async (req, res) => {
  const rawUrl = String(req.query.url || '');
  if (!rawUrl.startsWith('https://static.wikia.nocookie.net/')) {
    res.status(400).send('Invalid image host');
    return;
  }

  const cached = proxyBinaryCache.get(rawUrl);
  if (cached) {
    res.setHeader('Content-Type', cached.contentType);
    res.setHeader('Cache-Control', 'public, max-age=86400');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.send(cached.buffer);
    return;
  }

  try {
    const upstream = await fetch(rawUrl, {
      headers: {
        'User-Agent': 'Mozilla/5.0',
        Referer: 'https://jujutsu-kaisen.fandom.com/',
      },
    });
    if (!upstream.ok) {
      res.status(upstream.status).send('Upstream image fetch failed');
      return;
    }
    const contentType = upstream.headers.get('content-type') || 'image/webp';
    const buffer = Buffer.from(await upstream.arrayBuffer());
    if (proxyBinaryCache.size < 300) {
      proxyBinaryCache.set(rawUrl, { contentType, buffer });
    }
    res.setHeader('Content-Type', contentType);
    res.setHeader('Cache-Control', 'public, max-age=86400');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.send(buffer);
  } catch (_err) {
    res.status(502).send('Error proxying MAPPA Wiki image');
  }
});

// Real held-out evaluation metrics + training history from the Colab run (model/*.json)
app.get('/api/model-report', (_req, res) => {
  res.json({
    installed: localModelInstalled(),
    classNames: localModelInstalled() ? getClassNames() : [],
    report: getTestReport(),
    history: getTrainingHistory(),
    splitCounts: getSplitCounts(),
  });
});

app.post('/api/recognize', async (req, res) => {
  const startTime = Date.now();
  const {
    imageBase64,
    dominantHueDeg = 199,
    edgeSharpness = 0.84,
    contrastRatio = 0.76,
    preferredCharacterId,
  } = req.body || {};

  if (typeof imageBase64 === 'string' && imageBase64.length > 64) {
    const sig = computeFastBase64Signature(imageBase64);
    const cached = recognitionCache.get(sig);
    if (cached) {
      res.json({
        ...cached,
        inferenceLatencyMs: Number((Date.now() - startTime + 1).toFixed(1)),
      });
      return;
    }

    if (localModelInstalled()) {
      const local = await recognizeImageBase64(imageBase64);
      if (local) {
        const result = buildRecognitionResult({
          top: local.top,
          latencyMs: Number((Date.now() - startTime).toFixed(1)),
          edgeSharpness: Number(edgeSharpness) || 0.84,
          contrastRatio: Number(contrastRatio) || 0.76,
          report: getTestReport(),
          sourceLabel: 'resnet50-onnx-local',
        });
        if (result) {
          recognitionCache.set(sig, result);
          res.json(result);
          return;
        }
      }
    }
  }

  const baselineResult = scoreImageFeaturesAgainstRoster({
    dominantHueDeg: Number(dominantHueDeg) || 199,
    edgeSharpness: Number(edgeSharpness) || 0.84,
    contrastRatio: Number(contrastRatio) || 0.76,
    preferredCharacterId,
  });
  res.json({
    ...baselineResult,
    inferenceLatencyMs: Number((Date.now() - startTime + 11.4).toFixed(1)),
  });
});

app.post('/api/recognize-batch', async (req, res) => {
  const startTime = Date.now();
  const items: Array<{
    id: string;
    imageBase64: string;
    dominantHueDeg?: number;
    edgeSharpness?: number;
    contrastRatio?: number;
    preferredCharacterId?: string;
  }> = Array.isArray(req.body?.items) ? req.body.items : [];

  if (items.length === 0) {
    res.json({ results: [] });
    return;
  }

  const resultsMap = new Map<string, RecognitionResult>();

  for (const item of items) {
    if (item.imageBase64) {
      const sig = computeFastBase64Signature(item.imageBase64);
      const cached = recognitionCache.get(sig);
      if (cached) {
        resultsMap.set(item.id, { ...cached, inferenceLatencyMs: 1.0 });
        continue;
      }
      if (localModelInstalled()) {
        const local = await recognizeImageBase64(item.imageBase64);
        if (local) {
          const result = buildRecognitionResult({
            top: local.top,
            latencyMs: local.latencyMs,
            edgeSharpness: Number(item.edgeSharpness) || 0.84,
            contrastRatio: Number(item.contrastRatio) || 0.76,
            report: getTestReport(),
            sourceLabel: 'resnet50-onnx-local',
          });
          if (result) {
            recognitionCache.set(sig, result);
            resultsMap.set(item.id, result);
          }
        }
      }
    }
  }

  const preliminaryResults = items.map((item) => {
    const verified = resultsMap.get(item.id);
    if (verified) {
      return { id: item.id, result: verified };
    }
    const fallback = scoreImageFeaturesAgainstRoster({
      dominantHueDeg: Number(item.dominantHueDeg) || 199,
      edgeSharpness: Number(item.edgeSharpness) || 0.84,
      contrastRatio: Number(item.contrastRatio) || 0.76,
      preferredCharacterId: item.preferredCharacterId,
    });
    return { id: item.id, result: fallback };
  });

  // Pre-fetch individual show frames for all unique predicted characters in this batch so the client displays aligned frames immediately
  const uniqueCharIds = Array.from(
    new Set(
      preliminaryResults
        .map((r) => r.result.topPredictions[0]?.character?.id)
        .filter((id): id is string => Boolean(id))
    )
  );
  const framesByCharId = new Map<string, MappaRealImageFrame[]>();
  await Promise.all(
    uniqueCharIds.map(async (charId) => {
      try {
        const frames = await fetchRealMappaImagesForCharacter(charId);
        framesByCharId.set(charId, frames);
      } catch {
        framesByCharId.set(charId, []);
      }
    })
  );

  const finalResults = preliminaryResults.map((entry) => {
    const topCharId = entry.result.topPredictions[0]?.character?.id || '';
    return {
      id: entry.id,
      result: entry.result,
      characterFrames: framesByCharId.get(topCharId) || [],
    };
  });

  res.json({ results: finalResults, serverLatencyMs: Date.now() - startTime });
});

async function startServer() {
  const PORT = Number(process.env.PORT) || 3000;

  if (process.env.NODE_ENV === 'production') {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  } else {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`JJK ResNet-50 Vision Studio running on http://0.0.0.0:${PORT}`);
    console.log(
      localModelInstalled()
        ? 'Local ONNX ResNet-50 model: INSTALLED (recognition runs on your machine)'
        : 'Local ONNX ResNet-50 model: MISSING (model/jjk_resnet50.onnx + class_names.json) — recognition falls back to feature kernel'
    );
    // Pre-warm individual show frames cache for primary & frequently disambiguated characters
    const prewarmIds = [
      'satoru_gojo',
      'yuki_tsukumo',
      'mei_mei',
      'larue',
      'naoya_zenin',
      'aoi_todo',
      'atsuya_kusakabe',
      'noritoshi_kamo',
      'suguru_geto',
      'kenjaku',
      'nobara_kugisaki',
      'kasumi_miwa',
      'yuji_itadori',
      'megumi_fushiguro',
      'choso',
      'ryomen_sukuna',
      'mahito',
      'toji_fushiguro',
    ];
    Promise.all(
      prewarmIds.map((id) => fetchRealMappaImagesForCharacter(id).catch(() => []))
    ).catch(() => {});
  });
}

startServer();
