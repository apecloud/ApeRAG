'use client';

import { useState } from 'react';
import { CosmographSemanticMap } from './cosmograph-semantic-map';
import { CosmographTopology } from './cosmograph-topology';
import { GraphLabNav } from './graph-lab-nav';
import { DEFAULT_LAB_MODE, type LabMode } from './lab-modes';

export function CollectionGraphLab() {
  const [active, setActive] = useState<LabMode>(DEFAULT_LAB_MODE);
  return (
    <div className="flex h-full flex-1 gap-4 py-4">
      <GraphLabNav active={active} onChange={setActive} />
      <main className="flex-1 overflow-hidden">
        {active === 'cosmograph-topology' ? (
          <CosmographTopology />
        ) : (
          <CosmographSemanticMap />
        )}
      </main>
    </div>
  );
}
