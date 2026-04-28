'use client';

import { Skeleton } from '@/components/ui/skeleton';
import { getKnowledgeGraph } from '@/features/knowledge-graph/client-api';
import {
  toTopologyDataset,
  type CosmographDataset,
  type CosmographLink,
  type CosmographPoint,
} from '@/features/knowledge-graph/cosmograph-adapter';
import dynamic from 'next/dynamic';
import { useParams } from 'next/navigation';
import { useEffect, useMemo, useRef, useState } from 'react';

// Cosmograph is a WebGL renderer; importing it directly breaks the SSR
// pass with `window is not defined`. The dynamic import keeps the
// bundle out of the server graph.
const Cosmograph = dynamic(
  () => import('@cosmograph/react').then((m) => ({ default: m.Cosmograph })),
  { ssr: false },
);

type FrameStat = { renderMs?: number; pointCount: number; linkCount: number };

export function CosmographTopology() {
  const params = useParams<{ collectionId: string }>();
  const collectionId = params?.collectionId ?? '';
  const [dataset, setDataset] = useState<CosmographDataset | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<FrameStat | null>(null);
  const [selected, setSelected] = useState<CosmographPoint | null>(null);
  const renderStartRef = useRef<number>(0);

  useEffect(() => {
    if (!collectionId) return;
    let cancelled = false;
    setError(null);
    setDataset(null);
    getKnowledgeGraph(collectionId, {
      label: '*',
      maxNodes: 1000,
      maxDepth: 3,
    })
      .then((graph) => {
        if (cancelled) return;
        if (!graph) {
          setError('Empty graph response');
          return;
        }
        renderStartRef.current = performance.now();
        const next = toTopologyDataset(graph);
        setDataset(next);
        setStats({
          pointCount: next.points.length,
          linkCount: next.links.length,
        });
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [collectionId]);

  const cosmographConfig = useMemo(() => {
    if (!dataset) return null;
    return {
      points: dataset.points,
      links: dataset.links,
      pointIdBy: 'id',
      pointLabelBy: 'label',
      pointClusterBy: 'cluster',
      pointColorBy: 'cluster',
      linkSourceBy: 'source',
      linkTargetBy: 'target',
      backgroundColor: 'transparent',
      enableSimulation: true,
      simulationGravity: 0.25,
      simulationRepulsion: 1.0,
      pointSize: 6,
      onPointClick: (index: number) => {
        const point = dataset.points[index];
        if (point) setSelected(point);
      },
    } as const;
  }, [dataset]);

  if (error) {
    return (
      <div className="text-destructive flex h-full items-center justify-center text-sm">
        {error}
      </div>
    );
  }

  if (!cosmographConfig) {
    return (
      <div className="grid h-full grid-cols-[1fr_18rem] gap-4">
        <Skeleton className="h-full w-full" />
        <Skeleton className="h-full w-full" />
      </div>
    );
  }

  return (
    <div className="grid h-full grid-cols-[1fr_18rem] gap-4">
      <div className="bg-card relative flex h-full overflow-hidden rounded-lg border">
        <Cosmograph
          {...cosmographConfig}
          onMount={() => {
            const ms = performance.now() - renderStartRef.current;
            setStats((prev) =>
              prev
                ? { ...prev, renderMs: ms }
                : { pointCount: 0, linkCount: 0, renderMs: ms },
            );
          }}
          style={{ width: '100%', height: '100%' }}
        />
      </div>
      <DetailPanel
        stats={stats}
        clusterLabels={dataset?.clusterLabels ?? {}}
        selected={selected}
        links={dataset?.links ?? []}
      />
    </div>
  );
}

function DetailPanel({
  stats,
  clusterLabels,
  selected,
  links,
}: {
  stats: FrameStat | null;
  clusterLabels: Record<number, string>;
  selected: CosmographPoint | null;
  links: CosmographLink[];
}) {
  const neighbours = useMemo(() => {
    if (!selected) return [] as Array<{ direction: 'in' | 'out'; other: string }>;
    const out: Array<{ direction: 'in' | 'out'; other: string }> = [];
    for (const link of links) {
      if (link.source === selected.id) {
        out.push({ direction: 'out', other: link.target });
      } else if (link.target === selected.id) {
        out.push({ direction: 'in', other: link.source });
      }
    }
    return out.slice(0, 20);
  }, [selected, links]);

  return (
    <aside className="bg-card flex h-full flex-col gap-3 overflow-y-auto rounded-lg border p-4">
      <section>
        <h3 className="text-sm font-medium">Render</h3>
        <dl className="text-muted-foreground mt-1 grid grid-cols-2 gap-y-1 text-xs">
          <dt>points</dt>
          <dd className="text-foreground">{stats?.pointCount ?? '–'}</dd>
          <dt>links</dt>
          <dd className="text-foreground">{stats?.linkCount ?? '–'}</dd>
          <dt>first frame</dt>
          <dd className="text-foreground">
            {stats?.renderMs ? `${stats.renderMs.toFixed(0)} ms` : '–'}
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
            {selected.entity_type && (
              <div>
                <span className="text-muted-foreground">type: </span>
                <span>{selected.entity_type}</span>
              </div>
            )}
            {selected.description && (
              <p className="text-muted-foreground mt-1 line-clamp-6">
                {selected.description}
              </p>
            )}
            {neighbours.length > 0 && (
              <div className="mt-1">
                <div className="text-muted-foreground">neighbours</div>
                <ul className="font-mono">
                  {neighbours.map((n, i) => (
                    <li key={`${n.direction}-${n.other}-${i}`}>
                      {n.direction === 'in' ? '←' : '→'} {n.other}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ) : (
          <p className="text-muted-foreground mt-1 text-xs">
            Click a node in the canvas.
          </p>
        )}
      </section>
    </aside>
  );
}
