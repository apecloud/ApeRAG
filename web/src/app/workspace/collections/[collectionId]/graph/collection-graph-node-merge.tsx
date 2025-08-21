'use client';

import { MergeSuggestionsResponse } from '@/api';

import {
  Drawer,
  DrawerContent,
  DrawerHeader,
  DrawerTitle,
} from '@/components/ui/drawer';

export const CollectionGraphNodeMerge = ({
  dataSource,
  open,
  onClose,
  onSelectNode,
  onRefresh,
}: {
  dataSource: MergeSuggestionsResponse;
  open: boolean;
  onClose: () => void;
  onSelectNode: (id: string) => void;
  onRefresh: () => void;
}) => {
  return (
    <Drawer
      direction="right"
      open={open}
      onOpenChange={onClose}
      handleOnly={true}
    >
      <DrawerContent className="flex sm:min-w-lg md:min-w-xl lg:min-w-2xl">
        <DrawerHeader>
          <DrawerTitle>Merge suggestions</DrawerTitle>
        </DrawerHeader>
        <div className="flex-1 overflow-auto p-4 select-text"></div>
      </DrawerContent>
    </Drawer>
  );
};
