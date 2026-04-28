'use client';

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
import { handleSuggestionAction as dispatchSuggestionAction } from '@/features/knowledge-graph/client-api';
import type {
  MergeSuggestionItem,
  MergeSuggestionsResponse,
  MergeSuggestionStatus,
  SuggestionAction,
} from '@/features/knowledge-graph/types';
import { Check, LoaderCircle, Sparkles, X } from 'lucide-react';
import { useTranslations } from 'next-intl';
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
    useState<{ [key in SuggestionAction]: boolean }>();
  const page_graph = useTranslations('page_graph');
  const handleSuggestionAction = useCallback(
    async (action: SuggestionAction) => {
      setLoading({
        accept: action === 'accept',
        reject: action === 'reject',
      });
      const res = await dispatchSuggestionAction(item.collection_id, item.id, {
        action,
        target_entity_data: item.suggested_target_entity,
      });
      if (res?.status === 'success' && action === 'reject') {
        await afterRejectMergeSuggestion();
      }
      if (res?.status === 'success' && action === 'accept') {
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
    <div className="bg-card hover:bg-accent/70 flex flex-col gap-2 rounded-xl border px-4 py-3">
      <div className="flex flex-row items-center justify-between">
        <div
          className="hover:text-primary cursor-pointer font-serif text-base font-normal tracking-tight"
          onClick={() => onSelectNode(item.suggested_target_entity.entity_name)}
        >
          {item.suggested_target_entity.entity_name}
        </div>
        <div className="flex flex-row items-center gap-2">
          <div className="text-muted-foreground flex flex-row items-center gap-1 font-mono text-xs tabular-nums">
            <Sparkles className="size-3.5" />
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
              <TooltipContent>{page_graph('merge_accept')}</TooltipContent>
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
              <TooltipContent>{page_graph('merge_reject')}</TooltipContent>
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
    useState<MergeSuggestionStatus>('PENDING');
  const page_graph = useTranslations('page_graph');
  const suggestions = Array.isArray(dataSource.suggestions)
    ? dataSource.suggestions
    : [];
  const filteredSuggestions = suggestions.filter(
    (suggestion) => suggestion.status === activeStatus,
  );

  return (
    <Drawer
      direction="right"
      open={open}
      onOpenChange={onClose}
      handleOnly={true}
    >
      <DrawerContent className="sm:lg lg:min-w-2lg flex md:min-w-xl">
        <DrawerHeader className="flex flex-row items-center justify-between border-b">
          <DrawerTitle className="font-serif text-xl font-normal tracking-tight">
            {page_graph('merge_suggestions')}
          </DrawerTitle>
          <Tabs
            defaultValue={activeStatus}
            onValueChange={(v: string) =>
              setActiveStatus(v as MergeSuggestionStatus)
            }
          >
            <TabsList>
              <TabsTrigger value="PENDING">
                {page_graph('merge_pending')}
              </TabsTrigger>
              <TabsTrigger value="ACCEPTED">
                {page_graph('merge_accepted')}
              </TabsTrigger>
              <TabsTrigger value="REJECTED">
                {page_graph('merge_rejected')}
              </TabsTrigger>
            </TabsList>
          </Tabs>
        </DrawerHeader>
        <div className="flex flex-1 flex-col gap-2 overflow-auto p-2 select-text">
          {filteredSuggestions.length === 0 ? (
            <div className="text-muted-foreground flex flex-1 items-center justify-center px-6 text-center text-sm">
              {page_graph('no_nodes_found')}
            </div>
          ) : (
            filteredSuggestions.map((suggestion) => {
              return (
                <SuggestionItem
                  key={suggestion.id}
                  item={suggestion}
                  onSelectNode={onSelectNode}
                  afterRejectMergeSuggestion={onRefresh}
                  afterAcceptMergeSuggestion={onRefresh}
                />
              );
            })
          )}
        </div>
      </DrawerContent>
    </Drawer>
  );
};
