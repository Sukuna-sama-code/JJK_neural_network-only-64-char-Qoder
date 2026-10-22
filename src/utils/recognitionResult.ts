import {
  JJK_MAPPA_ROSTER,
  RecognitionPrediction,
  RecognitionResult,
} from './jjkClassifierLogic';

export interface TestReportShape {
  overall: { top1: number; top5: number; test_images: number };
  per_class: Record<string, { correct: number; total: number; acc: number | null }>;
  confused?: Array<[string, string, number]>;
}

export function buildRecognitionResult(params: {
  top: Array<{ id: string; probability: number }>;
  latencyMs: number;
  edgeSharpness: number;
  contrastRatio: number;
  report: TestReportShape | null;
  sourceLabel: RecognitionResult['analysisSource'];
}): RecognitionResult | null {
  const { top, latencyMs, edgeSharpness, contrastRatio, report, sourceLabel } = params;
  const topPredictions: RecognitionPrediction[] = top
    .map((p) => {
      const character = JJK_MAPPA_ROSTER.find((c) => c.id === p.id);
      if (!character) return null;
      return {
        character,
        probability: p.probability,
        cosineSimilarity: p.probability,
        matchedFeatures: character.visualDescriptors.slice(0, 2),
      };
    })
    .filter((p): p is RecognitionPrediction => Boolean(p));
  if (topPredictions.length === 0) return null;

  const winner = topPredictions[0];
  const runnerUp = topPredictions[1];
  const perClass = report?.per_class?.[winner.character.id] || null;
  const top1Pct = (top[0]?.probability ?? 0) * 100;
  const marginPts = ((top[0]?.probability ?? 0) - (top[1]?.probability ?? 0)) * 100;

  const statsBase = report?.overall
    ? `test top-1 ${(report.overall.top1 * 100).toFixed(1)}%, top-5 ${(report.overall.top5 * 100).toFixed(
        1
      )}% on ${report.overall.test_images} held-out wiki frames`
    : 'softmax over all 64 classes';

  return {
    topPredictions,
    bottleneckNorm: Number(top1Pct.toFixed(1)),
    layer4ActivationPeak: Number(marginPts.toFixed(1)),
    inferenceLatencyMs: latencyMs,
    featureSummary: {
      dominantColors: winner.character.paletteHex,
      edgeSharpness: Number(edgeSharpness.toFixed(3)),
      mappaLineworkScore: perClass ? perClass.acc : null,
      auraSignature: winner.character.cursedTechnique,
    },
    analysisSource: sourceLabel,
    visualReasoning: `${sourceLabel === 'resnet50-onnx-local' ? 'Local ONNX ResNet-50' : 'ONNX ResNet-50'} (${statsBase}) — softmax ${top1Pct.toFixed(1)}% for ${winner.character.name}, ${marginPts.toFixed(1)} pts ahead of ${runnerUp ? runnerUp.character.name : 'the runner-up'}.`,
    lowSupport: perClass ? perClass.total < 10 : undefined,
    perClassTestAcc: perClass ? perClass.acc : null,
    modelStats: report?.overall
      ? { top1: report.overall.top1, top5: report.overall.top5, testImages: report.overall.test_images }
      : undefined,
  };
}
