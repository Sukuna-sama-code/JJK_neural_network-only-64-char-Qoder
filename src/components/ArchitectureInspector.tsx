import React, { useEffect, useState } from 'react';
import { Layers, Activity } from 'lucide-react';
import { getModelReport } from '../utils/appBackend';

interface ResNetStageSpec {
  id: string;
  name: string;
  blocks: string;
  kernelSpec: string;
  outputShape: string;
  paramCount: string;
  roleInMappaJJK: string;
}

const RESNET50_STAGES: ResNetStageSpec[] = [
  {
    id: 'stem',
    name: 'Stage 0: Conv1 + MaxPool Stem',
    blocks: '7×7 Conv (stride 2) + BN + ReLU + 3×3 MaxPool',
    kernelSpec: '64 filters · 7×7 kernel',
    outputShape: '[B, 64, 56, 56]',
    paramCount: '9,536',
    roleInMappaJJK:
      'Extracts high-contrast MAPPA cel-shading ink lines, facial stitch contours (Mahito, Geto), and hair edge gradients.',
  },
  {
    id: 'layer1',
    name: 'Stage 1: Layer1 Bottleneck ×3',
    blocks: '3 × Bottleneck [1×1 (64) → 3×3 (64) → 1×1 (256)]',
    kernelSpec: '256 output channels · stride 1',
    outputShape: '[B, 256, 56, 56]',
    paramCount: '215,808',
    roleInMappaJJK:
      'Captures fine local textures: Sukuna facial tattoo bands, Inumaki Snake & Fangs sigils, and Gojo Six Eyes iris luminance.',
  },
  {
    id: 'layer2',
    name: 'Stage 2: Layer2 Bottleneck ×4',
    blocks: '4 × Bottleneck [1×1 (128) → 3×3 (128) → 1×1 (512)]',
    kernelSpec: '512 output channels · stride 2',
    outputShape: '[B, 512, 28, 28]',
    paramCount: '1,219,584',
    roleInMappaJJK:
      'Encodes mid-scale structural parts: spiky hair geometry (Gojo vs Yuji vs Megumi), eyewear (Nanami goggles, Maki glasses), and collar silhouettes.',
  },
  {
    id: 'layer3',
    name: 'Stage 3: Layer3 Bottleneck ×6',
    blocks: '6 × Bottleneck [1×1 (256) → 3×3 (256) → 1×1 (1024)]',
    kernelSpec: '1024 output channels · stride 2',
    outputShape: '[B, 1024, 14, 14]',
    paramCount: '7,098,368',
    roleInMappaJJK:
      'Integrates composite character anatomy & cursed tool props (Nobara hammer, Toji Inverted Spear of Heaven, Jogo volcano head).',
  },
  {
    id: 'layer4',
    name: 'Stage 4: Layer4 Bottleneck ×3',
    blocks: '3 × Bottleneck [1×1 (512) → 3×3 (512) → 1×1 (2048)]',
    kernelSpec: '2048 output channels · stride 2',
    outputShape: '[B, 2048, 7, 7]',
    paramCount: '14,964,736',
    roleInMappaJJK:
      'High-dimensional semantic representation combining full character identity, cursed energy aura chromaticity, and domain lighting.',
  },
  {
    id: 'head',
    name: 'Stage 5: AvgPool + Dropout FC Head',
    blocks:
      'AdaptiveAvgPool2d(1) → Dropout(0.35) → Linear(2048→512) → ReLU → Dropout(0.25) → Linear(512→64)',
    kernelSpec: '2048 → 512 → 64 classes (Softmax)',
    outputShape: '[B, 64]',
    paramCount: '1,081,920',
    roleInMappaJJK:
      'Softmax classifier over the 2048-d bottleneck identity vector separates look-alike pairs (Yuji vs Sukuna; Suguru Geto vs Kenjaku; Maki vs Mai Zenin; Megumi vs Toji).',
  },
];

interface HistoryEpoch {
  epoch: number;
  trainLoss: number;
  trainAcc: number;
  valLoss: number;
  valAcc: number;
  valTop5: number;
}

interface TrainingHistory {
  setup: Record<string, unknown>;
  best: { epoch: number; valTop1: number };
  test: { top1: number; top5: number; images: number };
  epochs_history: HistoryEpoch[];
}

export const ArchitectureInspector: React.FC = () => {
  const [selectedStage, setSelectedStage] = useState<ResNetStageSpec>(RESNET50_STAGES[4]);
  const [history, setHistory] = useState<TrainingHistory | null>(null);
  const [inspectedEpochIdx, setInspectedEpochIdx] = useState<number>(0);

  useEffect(() => {
    getModelReport()
      .then((data) => {
        const h = data?.history as TrainingHistory | null;
        if (h?.epochs_history?.length) {
          setHistory(h);
          setInspectedEpochIdx(
            Math.max(0, Math.min(h.epochs_history.length - 1, (h.best?.epoch ?? 1) - 1))
          );
        }
      })
      .catch(() => {});
  }, []);

  const curvePoints = history?.epochs_history ?? [];
  const activePoint =
    curvePoints[Math.min(curvePoints.length - 1, Math.max(0, inspectedEpochIdx))] ||
    curvePoints[curvePoints.length - 1];
  const setup = (history?.setup ?? {}) as Record<string, unknown>;
  const totalEpochs = Number(setup.epochs ?? curvePoints.length);

  // SVG chart dimensions
  const width = 680;
  const height = 240;
  const padLeft = 44;
  const padRight = 24;
  const padTop = 20;
  const padBottom = 32;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;

  const toX = (i: number) => padLeft + (i / Math.max(1, curvePoints.length - 1)) * plotW;
  const toYAcc = (acc: number) => padTop + (1 - acc) * plotH;
  const toYLoss = (loss: number) => padTop + Math.min(1, Math.max(0, 1 - loss / 3.8)) * plotH;

  const accPath = curvePoints
    .map((pt, i) => `${i === 0 ? 'M' : 'L'} ${toX(i).toFixed(1)} ${toYAcc(pt.valAcc).toFixed(1)}`)
    .join(' ');
  const top5Path = curvePoints
    .map((pt, i) => `${i === 0 ? 'M' : 'L'} ${toX(i).toFixed(1)} ${toYAcc(pt.valTop5).toFixed(1)}`)
    .join(' ');
  const lossPath = curvePoints
    .map((pt, i) => `${i === 0 ? 'M' : 'L'} ${toX(i).toFixed(1)} ${toYLoss(pt.valLoss).toFixed(1)}`)
    .join(' ');

  return (
    <section aria-labelledby="arch-heading" className="space-y-8">
      <div className="space-y-2 pb-6 border-b border-slate-800 max-w-3xl">
        <p className="text-xs font-mono text-cyan-400 tracking-wide">
          RESNET-50 BOTTLENECK TOPOLOGY · [3, 4, 6, 3] RESIDUAL BLOCKS · 64-CLASS SOFTMAX FC HEAD
        </p>
        <h1
          id="arch-heading"
          className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-100"
        >
          ResNet-50 Backbone Architecture & Convergence Telemetry
        </h1>
        <p className="text-sm text-slate-400 leading-relaxed">
          Explore how the 50-layer Residual Network extracts hierarchical anime features—from
          early 7×7 cel-shading edge filters to 2048-channel semantic bottleneck maps—and review
          the real fine-tuning convergence across all 64 JJK Wiki MAPPA character classes.
        </p>
      </div>

      {/* Interactive ResNet-50 Stage Pipeline Selector */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        <div className="lg:col-span-5 bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
              <Layers className="w-4 h-4 text-cyan-400" aria-hidden="true" />
              ResNet-50 Hierarchical Stages
            </h2>
            <span className="text-xs font-mono text-slate-400">Click Stage to Inspect</span>
          </div>

          <div className="space-y-2">
            {RESNET50_STAGES.map((stage, idx) => {
              const isSelected = selectedStage.id === stage.id;
              const isFrozen = idx <= 2;
              return (
                <button
                  key={stage.id}
                  type="button"
                  onClick={() => setSelectedStage(stage)}
                  className={`w-full p-3 text-left rounded-lg border transition-colors ${
                    isSelected
                      ? 'bg-cyan-500/15 border-cyan-400 text-slate-100'
                      : 'bg-slate-950/70 border-slate-800 text-slate-300 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-semibold">{stage.name}</span>
                    <span className="text-[11px] font-mono text-cyan-400 tabular-nums">
                      {stage.outputShape}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-slate-400 mt-1 font-mono">
                    <span>{stage.kernelSpec}</span>
                    <span>{isFrozen ? 'Frozen (Transfer)' : 'Trainable Grad'}</span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Stage Detail & Convergence Curve (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          {/* Active Stage Specification Panel */}
          <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-800">
              <div>
                <h3 className="text-base font-bold text-slate-100">{selectedStage.name}</h3>
                <p className="text-xs font-mono text-cyan-400 mt-0.5">{selectedStage.blocks}</p>
              </div>
              <div className="text-right font-mono">
                <div className="text-xs text-slate-400">OUTPUT TENSOR</div>
                <div className="text-sm font-bold text-slate-100 tabular-nums">
                  {selectedStage.outputShape}
                </div>
              </div>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed">
              <strong className="text-slate-100">MAPPA Feature Role: </strong>
              {selectedStage.roleInMappaJJK}
            </p>

            <div className="grid grid-cols-3 gap-3 pt-1 font-mono text-xs tabular-nums">
              <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg">
                <div className="text-[11px] text-slate-400">STAGE PARAMETERS</div>
                <div className="text-slate-100 font-semibold mt-0.5">{selectedStage.paramCount}</div>
              </div>
              <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg">
                <div className="text-[11px] text-slate-400">SKIP CONNECTION</div>
                <div className="text-slate-100 font-semibold mt-0.5">Identity + 1×1 Proj</div>
              </div>
              <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg">
                <div className="text-[11px] text-slate-400">ACTIVATION</div>
                <div className="text-cyan-400 font-semibold mt-0.5">BatchNorm2d + ReLU</div>
              </div>
            </div>
          </div>

          {/* Interactive Training Curve Simulator */}
          <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-5">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-cyan-400" aria-hidden="true" />
                  ResNet-50 Fine-Tuning Convergence Curve (64 Classes)
                </h3>
                <p className="text-xs text-slate-400">
                  {history
                    ? `Real run · ${String(setup.trainImages ?? '—')} train / ${String(
                        setup.valImages ?? '—'
                      )} val MAPPA wiki frames · held-out test top-1 ${(
                        (history.test?.top1 ?? 0) * 100
                      ).toFixed(1)}% / top-5 ${((history.test?.top5 ?? 0) * 100).toFixed(1)}% (${
                        history.test?.images ?? 0
                      } imgs)`
                    : 'Loading training history…'}
                </p>
              </div>

              {activePoint && (
                <div className="flex items-center gap-4 text-xs font-mono tabular-nums">
                  <span className="text-cyan-400">
                    Top-1: {(activePoint.valAcc * 100).toFixed(2)}%
                  </span>
                  <span className="text-emerald-400">
                    Top-5: {(activePoint.valTop5 * 100).toFixed(2)}%
                  </span>
                  <span className="text-amber-400">
                    Val Loss: {activePoint.valLoss.toFixed(3)}
                  </span>
                </div>
              )}
            </div>

            {/* Actual Training Setup (read-only) */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 p-3.5 bg-slate-950 border border-slate-800 rounded-lg text-xs">
              <div>
                <div className="text-slate-400 font-medium mb-1">Frozen Backbone Stages</div>
                <div className="px-2.5 py-1.5 bg-slate-900 border border-slate-700 rounded text-slate-100 font-mono">
                  {String(setup.frozenStages ?? 'Conv1/BN1 + Layer1 + Layer2')}
                </div>
              </div>

              <div>
                <div className="text-slate-400 font-medium mb-1">Optimizer / Schedule</div>
                <div className="px-2.5 py-1.5 bg-slate-900 border border-slate-700 rounded text-slate-100 font-mono">
                  {String(setup.optimizer ?? 'AdamW')} · {String(setup.scheduler ?? 'Cosine')}
                </div>
              </div>

              <div>
                <label
                  htmlFor="epoch-scrubber"
                  className="flex items-center justify-between text-slate-400 font-medium mb-1"
                >
                  <span>Inspect Epoch</span>
                  <span className="font-mono text-cyan-400 tabular-nums">
                    #{activePoint?.epoch ?? '—'} / {totalEpochs}
                  </span>
                </label>
                <input
                  id="epoch-scrubber"
                  type="range"
                  min={0}
                  max={Math.max(0, curvePoints.length - 1)}
                  value={Math.min(inspectedEpochIdx, Math.max(0, curvePoints.length - 1))}
                  onChange={(e) => setInspectedEpochIdx(Number(e.target.value))}
                  disabled={!curvePoints.length}
                  className="w-full accent-cyan-400 cursor-pointer"
                />
              </div>
            </div>

            {/* Dual-Axis SVG Line Chart */}
            <div className="w-full overflow-x-auto">
              <svg
                viewBox={`0 0 ${width} ${height}`}
                className="w-full h-auto bg-slate-950 border border-slate-800 rounded-lg"
                role="img"
                aria-label="ResNet-50 training accuracy and validation loss convergence chart"
              >
                {/* Horizontal grid lines */}
                {[0.2, 0.4, 0.6, 0.8, 1.0].map((val) => {
                  const y = toYAcc(val);
                  return (
                    <g key={val}>
                      <line
                        x1={padLeft}
                        y1={y}
                        x2={width - padRight}
                        y2={y}
                        stroke="#1E293B"
                        strokeDasharray="3 3"
                      />
                      <text
                        x={padLeft - 8}
                        y={y + 4}
                        textAnchor="end"
                        className="fill-slate-500 text-[10px] font-mono"
                      >
                        {Math.round(val * 100)}%
                      </text>
                    </g>
                  );
                })}

                {/* Curves: Top-5 (Emerald dashed), Top-1 (Cyan solid), Val Loss (Amber) */}
                <path
                  d={top5Path}
                  fill="none"
                  stroke="#10B981"
                  strokeWidth={2}
                  strokeDasharray="5 3"
                />
                <path d={accPath} fill="none" stroke="#06B6D4" strokeWidth={2.5} />
                <path d={lossPath} fill="none" stroke="#F59E0B" strokeWidth={2} />

                {/* Active Epoch Crosshair */}
                {activePoint && (
                  <>
                    <line
                      x1={toX(inspectedEpochIdx)}
                      y1={padTop}
                      x2={toX(inspectedEpochIdx)}
                      y2={height - padBottom}
                      stroke="#94A3B8"
                      strokeWidth={1}
                    />
                    <circle
                      cx={toX(inspectedEpochIdx)}
                      cy={toYAcc(activePoint.valAcc)}
                      r={4.5}
                      fill="#06B6D4"
                    />
                    <circle
                      cx={toX(inspectedEpochIdx)}
                      cy={toYLoss(activePoint.valLoss)}
                      r={4}
                      fill="#F59E0B"
                    />
                  </>
                )}
              </svg>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
