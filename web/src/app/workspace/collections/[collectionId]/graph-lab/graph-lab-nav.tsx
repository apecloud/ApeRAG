'use client';

import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import type { LabMode } from './lab-modes';

type Props = {
  active: LabMode;
  onChange: (mode: LabMode) => void;
};

const MODES: Array<{
  id: LabMode;
  title: string;
  description: string;
}> = [
  {
    id: 'cosmograph-topology',
    title: 'Cosmograph · Topology',
    description: 'WebGL force-directed render of the live KG nodes/edges.',
  },
  {
    id: 'cosmograph-semantic',
    title: 'Cosmograph · Semantic Map',
    description:
      'Pre-computed (x, y) point cloud, synthetic fixture for FPS bench.',
  },
];

const EXTERNAL_LINKS: Array<{
  href: (collectionId: string) => string;
  title: string;
  description: string;
}> = [
  {
    href: (id) => `/workspace/collections/${id}/graph`,
    title: 'Current · /graph',
    description: 'Production page, react-force-graph-2d.',
  },
];

export function GraphLabNav({ active, onChange }: Props) {
  const params = useParams<{ collectionId: string }>();
  const collectionId = params?.collectionId ?? '';
  return (
    <aside className="bg-card flex w-72 shrink-0 flex-col gap-4 border-r p-4">
      <div>
        <div className="text-muted-foreground mb-2 text-[10px] tracking-widest uppercase">
          Embedded modes
        </div>
        <div className="flex flex-col gap-2">
          {MODES.map((mode) => (
            <Button
              key={mode.id}
              variant={mode.id === active ? 'default' : 'outline'}
              className={cn('h-auto justify-start whitespace-normal py-3 text-left')}
              onClick={() => onChange(mode.id)}
            >
              <div className="flex flex-col items-start gap-1">
                <span className="text-sm font-medium">{mode.title}</span>
                <span className="text-muted-foreground text-xs">
                  {mode.description}
                </span>
              </div>
            </Button>
          ))}
        </div>
      </div>
      <div className="border-t pt-4">
        <div className="text-muted-foreground mb-2 text-[10px] tracking-widest uppercase">
          External pages
        </div>
        <div className="flex flex-col gap-2">
          {EXTERNAL_LINKS.map((link) => (
            <Link
              key={link.title}
              href={link.href(collectionId)}
              className="hover:bg-accent rounded-md border px-3 py-2"
            >
              <div className="text-sm font-medium">{link.title}</div>
              <div className="text-muted-foreground text-xs">
                {link.description}
              </div>
            </Link>
          ))}
        </div>
      </div>
      <div className="border-t pt-4">
        <div className="text-muted-foreground mb-2 text-[10px] tracking-widest uppercase">
          POC notes
        </div>
        <p className="text-muted-foreground text-xs leading-relaxed">
          Cosmograph (CC-BY-NC-4.0) is wired here as an exploratory PoC
          only. Productionising the visual requires either a paid
          Cosmograph licence or a switch to the MIT-licensed cosmos.gl
          base library — neither is in scope for this PoC.
        </p>
      </div>
    </aside>
  );
}
