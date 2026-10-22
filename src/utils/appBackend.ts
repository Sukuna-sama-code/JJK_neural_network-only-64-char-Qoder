import {
  MappaRealImageFrame,
  RecognitionResult,
  scoreImageFeaturesAgainstRoster,
} from './jjkClassifierLogic';
import { TestReportShape, buildRecognitionResult } from './recognitionResult';
import {
  browserRecognize,
  getBrowserClassNames,
  getStaticReport,
} from './browserInference';
import { fetchRealMappaImagesForCharacterBrowser } from './wikiImages';

// One façade for every backend touchpoint. On the local dev server the real
// Express endpoints answer; on the published static site the identical data
// comes from bundled JSON, browser-side ONNX inference and the fandom API.

export interface ModelReportPayload {
  installed: boolean;
  classNames: string[];
  report: TestReportShape | null;
  history: unknown;
  splitCounts: Record<string, { train: number; val: number; test: number }> | null;
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

// ?backend=browser forces the static-site path even when the dev server is up,
// so the published behaviour can be verified locally before deployment.
const FORCE_BROWSER =
  typeof location !== 'undefined' &&
  new URLSearchParams(location.search).get('backend') === 'browser';

export async function getModelReport(): Promise<ModelReportPayload> {
  if (!FORCE_BROWSER) {
    try {
      const r = await fetch('/api/model-report');
      if (r.ok) {
        const j = await r.json();
        if (j && typeof j === 'object' && ('installed' in j || 'report' in j)) return j;
      }
    } catch {
      // fall through to static assets
    }
  }
  const [classNames, report, history, splitCounts] = await Promise.all([
    fetchJsonSafe<string[]>('/model/class_names.json'),
    fetchJsonSafe<TestReportShape>('/model/jjk_test_report.json?v=20261022'),
    fetchJsonSafe<unknown>('/model/training_history.json'),
    fetchJsonSafe<Record<string, { train: number; val: number; test: number }>>(
      '/model/jjk_split_counts.json?v=20261022'
    ),
  ]);
  return {
    installed: Array.isArray(classNames) && classNames.length > 0,
    classNames: Array.isArray(classNames) ? classNames : [],
    report,
    history,
    splitCounts,
  };
}

export async function getCharacterImages(characterId: string): Promise<MappaRealImageFrame[]> {
  if (!FORCE_BROWSER) {
    try {
      const r = await fetch(`/api/character-images/${encodeURIComponent(characterId)}`);
      if (r.ok) {
        const j = await r.json();
        if (j && Array.isArray(j.images) && j.images.length > 0) return j.images;
      }
    } catch {
      // fall through to direct fandom API
    }
  }
  return fetchRealMappaImagesForCharacterBrowser(characterId);
}

export interface RecognizeBatchItem {
  id: string;
  imageBase64: string;
  mimeType?: string;
  dominantHueDeg?: number;
  edgeSharpness?: number;
  contrastRatio?: number;
  preferredCharacterId?: string;
}

export interface RecognizeBatchResult {
  id: string;
  result: RecognitionResult;
  characterFrames?: MappaRealImageFrame[];
}

async function browserRecognizeBatch(
  items: RecognizeBatchItem[]
): Promise<RecognizeBatchResult[]> {
  const report = await getStaticReport();
  await getBrowserClassNames();

  const results: RecognizeBatchResult[] = [];
  for (const item of items) {
    let result: RecognitionResult | null = null;
    if (item.imageBase64 && item.imageBase64.length > 64) {
      try {
        const local = await browserRecognize(item.imageBase64);
        if (local) {
          result = buildRecognitionResult({
            top: local.top,
            latencyMs: local.latencyMs,
            edgeSharpness: Number(item.edgeSharpness) || 0.84,
            contrastRatio: Number(item.contrastRatio) || 0.76,
            report,
            sourceLabel: 'resnet50-onnx-local',
          });
        }
      } catch {
        result = null;
      }
    }
    if (!result) {
      result = scoreImageFeaturesAgainstRoster({
        dominantHueDeg: Number(item.dominantHueDeg) || 199,
        edgeSharpness: Number(item.edgeSharpness) || 0.84,
        contrastRatio: Number(item.contrastRatio) || 0.76,
        preferredCharacterId: item.preferredCharacterId,
      });
    }
    results.push({ id: item.id, result });
  }

  const uniqueCharIds = Array.from(
    new Set(
      results
        .map((r) => r.result.topPredictions[0]?.character?.id)
        .filter((id): id is string => Boolean(id))
    )
  );
  const framesByCharId = new Map<string, MappaRealImageFrame[]>();
  await Promise.all(
    uniqueCharIds.map(async (charId) => {
      try {
        framesByCharId.set(charId, await getCharacterImages(charId));
      } catch {
        framesByCharId.set(charId, []);
      }
    })
  );

  return results.map((entry) => {
    const topCharId = entry.result.topPredictions[0]?.character?.id || '';
    return { ...entry, characterFrames: framesByCharId.get(topCharId) || [] };
  });
}

export async function recognizeBatch(
  items: RecognizeBatchItem[]
): Promise<{ results: RecognizeBatchResult[] }> {
  if (!FORCE_BROWSER) {
    try {
      const r = await fetch('/api/recognize-batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ items }),
      });
      if (r.ok) {
        const j = await r.json();
        if (j && Array.isArray(j.results)) return j;
      }
    } catch {
      // fall through to browser inference
    }
  }
  return { results: await browserRecognizeBatch(items) };
}
