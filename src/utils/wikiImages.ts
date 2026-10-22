import {
  JJK_MAPPA_ROSTER,
  MappaRealImageFrame,
  getWikiSearchTitlesForCharacter,
} from './jjkClassifierLogic';

// Browser port of the dev server's /api/character-images handler.
// fandom MediaWiki API supports anonymous CORS via the origin=* parameter, and
// static.wikia.nocookie.net sends access-control-allow-origin: * — so the published
// static site can fetch frames and draw them to canvas without any backend.

const framesCache = new Map<string, MappaRealImageFrame[]>();

export async function fetchRealMappaImagesForCharacterBrowser(
  characterId: string
): Promise<MappaRealImageFrame[]> {
  if (framesCache.has(characterId)) {
    return framesCache.get(characterId)!;
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
      if (/rescues\s+the\s+jujutsu\s+sorcerers|tombs\s+of\s+the\s+star\s+corridor\s+rejecting/i.test(title)) {
        return true;
      }
    }
    return false;
  };

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

  if (char.canonicalPortraitUrl) {
    const cleanCanonical = char.canonicalPortraitUrl.split('/revision/')[0];
    collected.set(cleanCanonical, {
      id: `${characterId}-canonical-0`,
      title: `${char.name} (Official MAPPA Anime Keyframe)`,
      wikiFileUrl: cleanCanonical,
      proxyUrl: cleanCanonical,
      width: 1280,
      height: 720,
      category: 'Character Sheet / Portrait',
    });
  }

  try {
    await Promise.all(
      searchTitles.map(async (prefix) => {
        const apiUrl = `https://jujutsu-kaisen.fandom.com/api.php?action=query&list=allimages&aiprefix=${encodeURIComponent(
          prefix
        )}&ailimit=100&format=json&origin=*`;
        const resp = await fetch(apiUrl);
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
              proxyUrl: cleanUrl,
              width: w,
              height: h,
              category: classifyFrameCategory(readableTitle, w, h),
            });
          }
        }
      })
    );

    const joinedTitles = searchTitles
      .flatMap((t) => [t, `${t}/Image_Gallery`, `${t}/Synopsis`])
      .join('|');
    const galleryUrl = `https://jujutsu-kaisen.fandom.com/api.php?action=query&titles=${encodeURIComponent(
      joinedTitles
    )}&generator=images&gimlimit=500&prop=imageinfo&iiprop=url|dimensions&format=json&origin=*`;
    const galleryResp = await fetch(galleryUrl);
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
            proxyUrl: cleanUrl,
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
    framesCache.set(characterId, finalFrames);
  }
  return finalFrames;
}
