'use client';

import { Markdown } from '@/components/markdown';
import {
  Drawer,
  DrawerContent,
  DrawerHeader,
  DrawerTitle,
} from '@/components/ui/drawer';
import { ENTITY_PALETTE, entityTypeToPaletteKey } from '@/lib/design-tokens';
import type { GraphNode } from '@/features/knowledge-graph/types';
import { useTranslations } from 'next-intl';

export const CollectionGraphNodeDetail = ({
  open,
  node,
  onClose,
}: {
  open: boolean;
  node?: GraphNode;
  onClose: () => void;
}) => {
  const page_graph = useTranslations('page_graph');
  const entityType = node?.properties.entity_type || 'UNKNOWN';
  const entityColor = ENTITY_PALETTE[entityTypeToPaletteKey(entityType)];
  const entityLabel = node
    ? // @ts-expect-error dynamic i18n key
      page_graph(`entity_${entityType}`)
    : '';

  return (
    <Drawer
      direction="right"
      open={open}
      onOpenChange={onClose}
      handleOnly={true}
    >
      <DrawerContent className="flex sm:min-w-sm md:min-w-md lg:min-w-lg">
        <DrawerHeader className="gap-2 border-b">
          <div className="flex items-center gap-2">
            <span
              className="size-2 shrink-0 rounded-full"
              style={{ backgroundColor: entityColor }}
            />
            <span className="text-muted-foreground font-mono text-[10px] uppercase tracking-wider">
              {entityLabel}
            </span>
          </div>
          <DrawerTitle className="font-serif text-xl font-normal leading-tight tracking-tight">
            {node?.id}
          </DrawerTitle>
        </DrawerHeader>
        <div className="flex-1 overflow-auto p-4 text-sm leading-relaxed select-text">
          {node?.properties.description ? (
            <Markdown>{node.properties.description}</Markdown>
          ) : (
            <p className="text-muted-foreground italic">
              {page_graph('no_description')}
            </p>
          )}
        </div>
      </DrawerContent>
    </Drawer>
  );
};
