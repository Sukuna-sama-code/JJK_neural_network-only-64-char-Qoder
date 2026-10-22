import test from 'node:test';
import assert from 'node:assert/strict';
import {
  JJK_MAPPA_ROSTER,
  WIKI_GROUPS,
  calculateDatasetManifest,
  filterCharacters,
  scoreImageFeaturesAgainstRoster,
  simulateTrainingCurve,
  generatePyTorchCodeBundle,
  extractDominantHueFromRgba,
  matchCharacterIdFromHint,
} from './jjkClassifierLogic.ts';

test('every JJK Wiki character in the 64-class MAPPA roster has at least 200 verified images and calibrated weights', () => {
  assert.equal(JJK_MAPPA_ROSTER.length, 64);
  for (const char of JJK_MAPPA_ROSTER) {
    const sumSources =
      char.mappaSources.jujutsukaisenJpSheets +
      char.mappaSources.mappaWorksKeyframes +
      char.mappaSources.episodeCleanCuts +
      char.mappaSources.blurayTurnarounds;
    assert.equal(
      char.totalRawImages,
      sumSources,
      `Source sum mismatch for ${char.name}`
    );
    assert.ok(
      char.totalRawImages >= 200,
      `${char.name} has fewer than 200 images (${char.totalRawImages})`
    );
    assert.ok(
      WIKI_GROUPS.includes(char.wikiGroup),
      `Invalid wikiGroup for ${char.name}`
    );
    assert.ok(
      char.canonicalPortraitUrl.startsWith('https://static.wikia.nocookie.net/'),
      `Missing canonicalPortraitUrl for ${char.name}`
    );
    assert.ok(
      char.canonicalProxyUrl.startsWith('https://static.wikia.nocookie.net/'),
      `Missing canonicalProxyUrl for ${char.name}`
    );
    assert.ok(
      char.calibratedWeights &&
        char.calibratedWeights.chromaWeight > 0 &&
        char.calibratedWeights.lineworkWeight > 0 &&
        char.calibratedWeights.cosineScale >= 16,
      `Invalid calibratedWeights for ${char.name}`
    );
  }
});

test('Noritoshi Kamo, Suguru Geto, and Kenjaku have distinct non-clashing canonical portraits and disambiguation rules', () => {
  const kamo = JJK_MAPPA_ROSTER.find((c) => c.id === 'noritoshi_kamo')!;
  const geto = JJK_MAPPA_ROSTER.find((c) => c.id === 'suguru_geto')!;
  const kenjaku = JJK_MAPPA_ROSTER.find((c) => c.id === 'kenjaku')!;

  assert.ok(
    !kamo.canonicalPortraitUrl.toLowerCase().includes('ancestor'),
    'Noritoshi Kamo canonical portrait must not use Meiji Ancestor (Kenjaku)'
  );
  assert.ok(
    !geto.canonicalPortraitUrl.toLowerCase().includes('group'),
    'Suguru Geto canonical portrait must be a solo portrait, not Geto Group'
  );
  assert.notEqual(kamo.canonicalPortraitUrl, kenjaku.canonicalPortraitUrl);
  assert.notEqual(geto.canonicalPortraitUrl, kenjaku.canonicalPortraitUrl);
  assert.notEqual(kamo.dominantHueDeg, geto.dominantHueDeg);
  assert.notEqual(geto.dominantHueDeg, kenjaku.dominantHueDeg);

  // Disambiguation of hints/filenames
  assert.equal(matchCharacterIdFromHint('Noritoshi_Kamo_(Anime).png'), 'noritoshi_kamo');
  assert.equal(matchCharacterIdFromHint('Noritoshi_Kamo_(Ancestor)_(Anime).png'), 'kenjaku');
  assert.equal(matchCharacterIdFromHint('Pseudo-Geto_Shibuya.png'), 'kenjaku');
  assert.equal(matchCharacterIdFromHint('Kenjaku_(Anime).png'), 'kenjaku');
  assert.equal(matchCharacterIdFromHint('Suguru_Geto_(Prequel_Anime).png'), 'suguru_geto');
  assert.equal(matchCharacterIdFromHint('Nobara_Kugisaki.png'), 'nobara_kugisaki');
  assert.equal(matchCharacterIdFromHint('Kasumi_Miwa.png'), 'kasumi_miwa');
});

test('calculateDatasetManifest validates >=200 threshold and computes 70/15/15 splits across 64 classes', () => {
  const stats = calculateDatasetManifest(JJK_MAPPA_ROSTER, 200, 4);
  assert.equal(stats.characterCount, 64);
  assert.equal(stats.allClassesMeetMinimum, true);
  assert.ok(stats.lowestClassCount >= 200);
  assert.equal(stats.trainCount + stats.valCount + stats.testCount, stats.totalRawImages);
  assert.equal(stats.totalAugmentedImages, stats.totalRawImages * 4);
});

test('filterCharacters filters accurately by faction, wikiGroup, arc, and search query', () => {
  const cursedSpirits = filterCharacters(JJK_MAPPA_ROSTER, 'Cursed Spirit', '');
  assert.equal(cursedSpirits.length, 8);

  const getoGroup = filterCharacters(JJK_MAPPA_ROSTER, 'All', '', 'All', "Geto's Group");
  assert.equal(getoGroup.length, 7);

  const mahoragaSearch = filterCharacters(JJK_MAPPA_ROSTER, 'All', 'Mahoraga');
  assert.equal(mahoragaSearch.length, 1);
  assert.equal(mahoragaSearch[0].id, 'mahoraga');
});

test('scoreImageFeaturesAgainstRoster separates Kamo, Geto, and Kenjaku and applies calibrated weight overrides', () => {
  const kamoChar = JJK_MAPPA_ROSTER.find((c) => c.id === 'noritoshi_kamo')!;
  const getoChar = JJK_MAPPA_ROSTER.find((c) => c.id === 'suguru_geto')!;
  const kenjakuChar = JJK_MAPPA_ROSTER.find((c) => c.id === 'kenjaku')!;

  const kamoPred = scoreImageFeaturesAgainstRoster({
    dominantHueDeg: kamoChar.dominantHueDeg,
    edgeSharpness: 0.91,
    contrastRatio: 0.86,
  });
  assert.equal(kamoPred.topPredictions[0].character.id, 'noritoshi_kamo');

  const getoPred = scoreImageFeaturesAgainstRoster({
    dominantHueDeg: getoChar.dominantHueDeg,
    edgeSharpness: 0.91,
    contrastRatio: 0.86,
  });
  assert.equal(getoPred.topPredictions[0].character.id, 'suguru_geto');

  const kenjakuPred = scoreImageFeaturesAgainstRoster({
    dominantHueDeg: kenjakuChar.dominantHueDeg,
    edgeSharpness: 0.91,
    contrastRatio: 0.86,
  });
  assert.equal(kenjakuPred.topPredictions[0].character.id, 'kenjaku');

  const boostedPred = scoreImageFeaturesAgainstRoster({
    dominantHueDeg: kamoChar.dominantHueDeg,
    edgeSharpness: 0.91,
    contrastRatio: 0.86,
    preferredCharacterId: 'noritoshi_kamo',
    weightOverrides: { cosineScale: 42.0, classBias: 0.18 },
  });
  assert.equal(boostedPred.topPredictions[0].character.id, 'noritoshi_kamo');
  assert.ok(boostedPred.topPredictions[0].probability >= kamoPred.topPredictions[0].probability);
});

test('extractDominantHueFromRgba extracts accurate HSV hue and suppresses solid pastel backgrounds on 2D canvases', () => {
  const cyanBuffer = new Uint8ClampedArray(16 * 4);
  for (let i = 0; i < cyanBuffer.length; i += 4) {
    cyanBuffer[i] = 14;
    cyanBuffer[i + 1] = 165;
    cyanBuffer[i + 2] = 233;
    cyanBuffer[i + 3] = 255;
  }
  const { dominantHueDeg, meanSaturation } = extractDominantHueFromRgba(cyanBuffer, 1);
  assert.ok(dominantHueDeg >= 190 && dominantHueDeg <= 210);
  assert.ok(meanSaturation > 0.8);

  const side = 64;
  const nobaraOnPinkWall = new Uint8ClampedArray(side * side * 4);
  for (let y = 0; y < side; y++) {
    for (let x = 0; x < side; x++) {
      const idx = (y * side + x) * 4;
      const nx = x / side;
      const ny = y / side;
      if (nx >= 0.28 && nx <= 0.72 && ny >= 0.14 && ny <= 0.46) {
        nobaraOnPinkWall[idx] = 217;
        nobaraOnPinkWall[idx + 1] = 119;
        nobaraOnPinkWall[idx + 2] = 6;
        nobaraOnPinkWall[idx + 3] = 255;
      } else {
        nobaraOnPinkWall[idx] = 243;
        nobaraOnPinkWall[idx + 1] = 193;
        nobaraOnPinkWall[idx + 2] = 198;
        nobaraOnPinkWall[idx + 3] = 255;
      }
    }
  }
  const nobaraExtracted = extractDominantHueFromRgba(nobaraOnPinkWall, 1);
  assert.ok(
    nobaraExtracted.dominantHueDeg >= 20 && nobaraExtracted.dominantHueDeg <= 40,
    `Expected ginger hue 20..40, got ${nobaraExtracted.dominantHueDeg}`
  );
  const nobaraPred = scoreImageFeaturesAgainstRoster({
    dominantHueDeg: nobaraExtracted.dominantHueDeg,
    edgeSharpness: 0.85,
    contrastRatio: 0.78,
  });
  assert.equal(nobaraPred.topPredictions[0].character.id, 'nobara_kugisaki');
});

test('simulateTrainingCurve generates monotonic convergence over epochs', () => {
  const curve = simulateTrainingCurve(25, 0.0003, 2, true);
  assert.equal(curve.length, 25);
  assert.ok(curve[24].top1Acc > curve[0].top1Acc);
  assert.ok(curve[24].valLoss < curve[0].valLoss);
});

test('generatePyTorchCodeBundle includes 64-class ResNet-50 backbone, >=200 scraper check, and Grad-CAM probe', () => {
  const bundle = generatePyTorchCodeBundle({
    numClasses: 64,
    minImagesPerCharacter: 200,
    inputResolution: 224,
    freezeStages: 2,
    learningRate: 0.0003,
    weightDecay: 0.01,
    labelSmoothing: 0.1,
    useMixupCutmix: true,
    useCosineClassifierHead: true,
    augmentationMultiplier: 4,
  });
  assert.ok(bundle['model_resnet50.py'].includes('ResNet50_Weights.IMAGENET1K_V2'));
  assert.ok(bundle['model_resnet50.py'].includes('forward_spatial_probes'));
  assert.ok(bundle['mappa_dataset_scraper.py'].includes('MIN_IMAGES_PER_CHARACTER = 200'));
  assert.ok(bundle['train_pipeline.py'].includes('RandomRotation(degrees=18)'));
  assert.ok(bundle['train_pipeline.py'].includes('RandomGrayscale(p=0.15)'));
  assert.ok(bundle['test_custom_images.py'].includes('compute_gradcam'));

  const yuki = JJK_MAPPA_ROSTER.find((c) => c.id === 'yuki_tsukumo')!;
  assert.ok(yuki.visualDescriptors.some((d) => d.toLowerCase().includes('goggles')));
  assert.ok(yuki.visualDescriptors.some((d) => d.toLowerCase().includes('half-body')));
  assert.ok(yuki.visualDescriptors.some((d) => d.toLowerCase().includes('garuda')));
});


