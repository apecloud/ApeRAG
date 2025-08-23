'use client';

import {
  MergeSuggestionItem,
  MergeSuggestionItemStatusEnum,
  MergeSuggestionsResponse,
  SuggestionActionRequestActionEnum,
} from '@/api';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

import {
  Drawer,
  DrawerContent,
  DrawerHeader,
  DrawerTitle,
} from '@/components/ui/drawer';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { apiClient } from '@/lib/api/client';
import { Check, LoaderCircle, Sparkles, X } from 'lucide-react';
import { useCallback, useState } from 'react';

const SuggestionItem = ({
  item,
  onSelectNode,
  afterRejectMergeSuggestion,
  afterAcceptMergeSuggestion,
}: {
  item: MergeSuggestionItem;
  onSelectNode: (name: string) => void;
  afterRejectMergeSuggestion: () => void;
  afterAcceptMergeSuggestion: () => void;
}) => {
  const [loading, setLoading] =
    useState<{ [key in SuggestionActionRequestActionEnum]: boolean }>();

  const handleSuggestionAction = useCallback(
    async (action: SuggestionActionRequestActionEnum) => {
      setLoading({
        accept: action === 'accept',
        reject: action === 'reject',
      });
      const res =
        await apiClient.graphApi.collectionsCollectionIdGraphsMergeSuggestionsSuggestionIdActionPost(
          {
            suggestionId: item.id,
            collectionId: item.collection_id,
            suggestionActionRequest: {
              action,
              target_entity_data: item.suggested_target_entity,
            },
          },
        );
      if (res.data.status === 'success' && action === 'reject') {
        await afterRejectMergeSuggestion();
      }
      if (res.data.status === 'success' && action === 'accept') {
        await afterAcceptMergeSuggestion();
      }
      setLoading({
        accept: false,
        reject: false,
      });
    },
    [
      afterAcceptMergeSuggestion,
      afterRejectMergeSuggestion,
      item.collection_id,
      item.id,
      item.suggested_target_entity,
    ],
  );

  return (
    <div className="hover:bg-muted/50 flex flex-col gap-2 border-b px-4 py-2">
      <div className="flex flex-row items-center justify-between">
        <div
          className="cursor-pointer"
          onClick={() => onSelectNode(item.suggested_target_entity.entity_name)}
        >
          {item.suggested_target_entity.entity_name}
        </div>
        <div className="flex flex-row items-center gap-2">
          <div className="flex flex-row items-center gap-1 text-sm">
            <Sparkles className="text-muted-foreground size-4" />
            {item.confidence_score}
          </div>
          {item.status === 'PENDING' && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  size="icon"
                  variant="ghost"
                  className="cursor-pointer"
                  onClick={() => handleSuggestionAction('accept')}
                >
                  {loading?.accept ? (
                    <LoaderCircle className="animate-spin" />
                  ) : (
                    <Check className="text-green-600" />
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent>Accept</TooltipContent>
            </Tooltip>
          )}
          {item.status === 'PENDING' && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  size="icon"
                  variant="ghost"
                  className="cursor-pointer"
                  onClick={() => handleSuggestionAction('reject')}
                >
                  {loading?.reject ? (
                    <LoaderCircle className="animate-spin" />
                  ) : (
                    <X className="text-rose-600" />
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent>Reject</TooltipContent>
            </Tooltip>
          )}
        </div>
      </div>
      <div className="text-muted-foreground text-sm">{item.merge_reason}</div>
      <div className="flex flex-wrap gap-1">
        {item.entity_ids.map((entity) => (
          <Badge
            key={entity}
            variant="outline"
            className="cursor-pointer"
            onClick={() => onSelectNode(entity)}
          >
            {entity}
          </Badge>
        ))}
      </div>
    </div>
  );
};

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
  const [activeStatus, setActiveStatus] =
    useState<MergeSuggestionItemStatusEnum>('PENDING');

  return (
    <Drawer
      direction="right"
      open={open}
      onOpenChange={onClose}
      handleOnly={true}
    >
      <DrawerContent className="flex sm:min-w-sm md:min-w-md lg:min-w-lg">
        <DrawerHeader className="flex flex-row items-center justify-between border-b">
          <DrawerTitle>Merge suggestions</DrawerTitle>
          <Tabs
            defaultValue={activeStatus}
            onValueChange={(v: string) =>
              setActiveStatus(v as MergeSuggestionItemStatusEnum)
            }
          >
            <TabsList>
              <TabsTrigger value={MergeSuggestionItemStatusEnum.PENDING}>
                Pending
              </TabsTrigger>
              <TabsTrigger value={MergeSuggestionItemStatusEnum.ACCEPTED}>
                Accepted
              </TabsTrigger>
              <TabsTrigger value={MergeSuggestionItemStatusEnum.REJECTED}>
                Rejected
              </TabsTrigger>
            </TabsList>
          </Tabs>
        </DrawerHeader>
        <div className="flex-1 overflow-auto select-text">
          {dataSource.suggestions
            .filter((s) => s.status === activeStatus)
            .map((suggestion) => {
              return (
                <SuggestionItem
                  key={suggestion.id}
                  item={suggestion}
                  onSelectNode={onSelectNode}
                  afterRejectMergeSuggestion={onRefresh}
                  afterAcceptMergeSuggestion={onRefresh}
                />
              );
            })}
        </div>
      </DrawerContent>
    </Drawer>
  );
};
