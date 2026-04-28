'use client';

import { Button } from '@/components/ui/button';
import {
  buildSyntheticSemanticDataset,
  type CosmographDataset,
  type CosmographPoint,
} from '@/features/knowledge-graph/cosmograph-adapter';
import { cn } from '@/lib/utils';
import dynamic from 'next/dynamic';
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  SEMANTIC_SCALE_PRESETS,
  type SemanticScale,
} from './lab-modes';

const Cosmograph = dynamic(
  () => import('@cosmograph/react').then((m) => ({ default: m.Cosmograph })),
  { ssr: false },
);

type Bench = {
  scale: number;
  generateMs: number;
  renderMs: number | null;
};

export function CosmographSemanticMap() {
  const [scale, setScale] = useState<SemanticScale>(SEMANTIC_SCALE_PRESETS[0]);
  const [dataset, setDataset] = useState<CosmographDataset | null>(null);
  const [bench, setBench] = useState<Bench | null>(null);
  const [selected, setSelected] = useState<CosmographPoint | null>(null);
  const renderStartRef = useRef<number>(0);

  useEffect(() => {
    const generateStart = performance.now();
    const next = buildSyntheticSemanticDataset(scale, { clusterCount: 8 });
    const generateMs = performance.now() - generateStart;
    setDataset(next);
    renderStartRef.current = performance.now();
    setBench({ scale, generateMs, renderMs: null });
    setSelected(null);
  }, [scale]);

  const cosmographConfig = useMemo(() => {
    if (!dataset) return null;
    return {
      points: dataset.points,
      pointIdBy: 'id',
      pointLabelBy: 'label',
      pointClusterBy: 'cluster',
      pointColorBy: 'cluster',
      pointXBy: 'x',
      pointYBy: 'y',
      backgroundColor: 'transparent',
      // Semantic mode: positions are pre-baked, suppress the simulation.
      enableSimulation: false,
      pointSize: 4,
      onPointClick: (index: number) => {
        const point = dataset.points[index];
        if (point) setSelected(point);
      },
    } as const;
  }, [dataset]);

  return (
    <div className="grid h-full grid-cols-[1fr_18rem] gap-4">
      <div className="bg-card relative flex h-full flex-col overflow-hidden rounded-lg border">
        <div className="flex items-center gap-2 border-b px-3 py-2">
          <span className="text-muted-foreground text-xs">Synthetic scale:</span>
          {SEMANTIC_SCALE_PRESETS.map((preset) => (
            <Button
              key={preset}
              size="sm"
              variant={preset === scale ? 'default' : 'outline'}
              className={cn('h-7 px-2 text-xs')}
              onClick={() => setScale(preset)}
            >
              {preset.toLocaleString()}
            </Button>
          ))}
        </div>
        <div className="relative flex-1">
          {cosmographConfig && (
            <Cosmograph
              {...cosmographConfig}
              onMount={() => {
                const ms = performance.now() - renderStartRef.current;
                setBench((prev) =>
                  prev ? { ...prev, renderMs: ms } : prev,
                );
              }}
              style={{ width: '100%', height: '100%' }}
            />
          )}
        </div>
      </div>
      <SemanticDetail
        bench={bench}
        clusterLabels={dataset?.clusterLabels ?? {}}
        selected={selected}
      />
    </div>
  );
}

function SemanticDetail({
  bench,
  clusterLabels,
  selected,
}: {
  bench: Bench | null;
  clusterLabels: Record<number, string>;
  selected: CosmographPoint | null;
}) {
  return (
    <aside className="bg-card flex h-full flex-col gap-3 overflow-y-auto rounded-lg border p-4">
      <section>
        <h3 className="text-sm font-medium">Bench</h3>
        <dl className="text-muted-foreground mt-1 grid grid-cols-2 gap-y-1 text-xs">
          <dt>points</dt>
          <dd className="text-foreground">
            {bench ? bench.scale.toLocaleString() : '–'}
          </dd>
          <dt>generate</dt>
          <dd className="text-foreground">
            {bench ? `${bench.generateMs.toFixed(1)} ms` : '–'}
          </dd>
          <dt>first frame</dt>
          <dd className="text-foreground">
            {bench?.renderMs ? `${bench.renderMs.toFixed(0)} ms` : '–'}
          </dd>
        </dl>
      </section>
      <section>
        <h3 className="text-sm font-medium">Clusters</h3>
        <ul className="text-muted-foreground mt-1 max-h-32 overflow-y-auto text-xs">
          {Object.entries(clusterLabels).map(([id, label]) => (
            <li key={id}>{label}</li>
          ))}
        </ul>
      </section>
      <section>
        <h3 className="text-sm font-medium">Selected</h3>
        {selected ? (
          <div className="mt-1 flex flex-col gap-1 text-xs">
            <div>
              <span className="text-muted-foreground">id: </span>
              <span className="font-mono">{selected.id}</span>
            </div>
            <div>
              <span className="text-muted-foreground">label: </span>
              <span>{selected.label}</span>
            </div>
            <div>
              <span className="text-muted-foreground">cluster: </span>
              <span>{clusterLabels[selected.cluster] ?? selected.cluster}</span>
            </div>
            {typeof selected.x === 'number' && typeof selected.y === 'number' && (
              <div>
                <span className="text-muted-foreground">coords: </span>
                <span className="font-mono">
                  ({selected.x.toFixed(3)}, {selected.y.toFixed(3)})
                </span>
              </div>
            )}
            {selected.description && (
              <p className="text-muted-foreground mt-1 line-clamp-6">
                {selected.description}
              </p>
            )}
          </div>
        ) : (
          <p className="text-muted-foreground mt-1 text-xs">
            Click a point in the canvas.
          </p>
        )}
      </section>
    </aside>
  );
}
