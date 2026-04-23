'use client';

import { Markdown } from '@/components/markdown';
import {
  Drawer,
  DrawerContent,
  DrawerHeader,
  DrawerTitle,
} from '@/components/ui/drawer';
import type { GraphNode } from '@/features/knowledge-graph/types';

export const CollectionGraphNodeDetail = ({
  open,
  node,
  onClose,
}: {
  open: boolean;
  node?: GraphNode;
  onClose: () => void;
}) => {
  return (
    <Drawer
      direction="right"
      open={open}
      onOpenChange={onClose}
      handleOnly={true}
    >
      <DrawerContent className="flex sm:min-w-sm md:min-w-md lg:min-w-lg">
        <DrawerHeader>
          <DrawerTitle>{node?.id}</DrawerTitle>
        </DrawerHeader>
        <div className="flex-1 overflow-auto p-4 select-text">
          <Markdown>{node?.properties.description ?? undefined}</Markdown>
        </div>
      </DrawerContent>
    </Drawer>
  );
};
