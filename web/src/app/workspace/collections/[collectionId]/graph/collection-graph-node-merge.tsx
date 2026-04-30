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
import { Ban, Check, LoaderCircle, RefreshCw, Sparkles, X } from 'lucide-react';
import { useTranslations } from 'next-intl';
import { useCallback, useMemo, useState } from 'react';

const STATUS_FILTERS: MergeSuggestionStatus[] = [
  'PENDING',
  'APPLY_PENDING',
  'APPLYING',
  'APPLIED',
  'APPLY_FAILED',
  'REJECTED',
  'DISMISSED',
  'ACCEPTED',
  'EXPIRED',
  'SUPERSEDED',
];

const STATUS_LABEL_KEYS = {
  PENDING: 'merge_status_pending',
  APPLY_PENDING: 'merge_status_apply_pending',
  APPLYING: 'merge_status_applying',
  APPLIED: 'merge_status_applied',
  APPLY_FAILED: 'merge_status_apply_failed',
  ACCEPTED: 'merge_status_accepted_legacy',
  REJECTED: 'merge_status_rejected',
  DISMISSED: 'merge_status_dismissed',
  EXPIRED: 'merge_status_expired',
  SUPERSEDED: 'merge_status_superseded',
} as const satisfies Record<MergeSuggestionStatus, string>;

const statusTone = (status: MergeSuggestionStatus) => {
  if (status === 'PENDING')
    return 'border-amber-300 bg-amber-50 text-amber-700';
  if (status === 'APPLY_PENDING' || status === 'APPLYING') {
    return 'border-sky-300 bg-sky-50 text-sky-700';
  }
  if (status === 'APPLIED' || status === 'ACCEPTED') {
    return 'border-green-300 bg-green-50 text-green-700';
  }
  if (status === 'APPLY_FAILED') {
    return 'border-rose-300 bg-rose-50 text-rose-700';
  }
  return 'border-muted bg-muted text-muted-foreground';
};

const uniqueStrings = (items: Array<string | null | undefined>) =>
  Array.from(
    new Set(items.map((item) => item?.trim()).filter(Boolean)),
  ) as string[];

const resolveTarget = (item: MergeSuggestionItem) => {
  const targetId =
    item.target_entity_id || item.suggested_target_entity.entity_name;
  const targetEntity = item.entities?.find(
    (entity) =>
      entity.entity_id === targetId ||
      entity.entity_name === targetId ||
      entity.entity_name === item.suggested_target_entity.entity_name,
  );

  return {
    entity_name:
      targetEntity?.entity_name ||
      item.suggested_target_entity.entity_name ||
      targetId,
    entity_type:
      targetEntity?.entity_type ||
      item.suggested_target_entity.entity_type ||
      '',
  };
};

const resolveEntityLabel = (item: MergeSuggestionItem, entityId: string) => {
  const entity = item.entities?.find((entry) => entry.entity_id === entityId);
  return entity?.entity_name || entityId;
};

const resolveObservedTypes = (item: MergeSuggestionItem) =>
  uniqueStrings(item.entities?.map((entity) => entity.entity_type) || []);

const resolveAffectedDocCount = (item: MergeSuggestionItem) => {
  return uniqueStrings(item.evidence_refs?.map((ref) => ref.document_id) || [])
    .length;
};

const SuggestionItem = ({
  item,
  onSelectNode,
  afterRejectMergeSuggestion,
  afterAcceptMergeSuggestion,
  afterDismissMergeSuggestion,
}: {
  item: MergeSuggestionItem;
  onSelectNode: (name: string) => void;
  afterRejectMergeSuggestion: () => void;
  afterAcceptMergeSuggestion: () => void;
  afterDismissMergeSuggestion: () => void;
}) => {
  const [loading, setLoading] =
    useState<Partial<Record<SuggestionAction, boolean>>>();
  const page_graph = useTranslations('page_graph');
  const target = resolveTarget(item);
  const observedTypes = resolveObservedTypes(item);
  const typeConflict = observedTypes.filter(Boolean).length > 1;
  const affectedDocCount = resolveAffectedDocCount(item);
  const canAct = item.status === 'PENDING';

  const handleSuggestionAction = useCallback(
    async (action: SuggestionAction) => {
      setLoading((value) => ({ ...value, [action]: true }));
      try {
        const res = await dispatchSuggestionAction(
          item.collection_id,
          item.id,
          { action },
        );
        if (res?.status === 'success' && action === 'reject') {
          await afterRejectMergeSuggestion();
        }
        if (res?.status === 'success' && action === 'accept') {
          await afterAcceptMergeSuggestion();
        }
        if (res?.status === 'success' && action === 'dismiss') {
          await afterDismissMergeSuggestion();
        }
      } finally {
        setLoading((value) => ({ ...value, [action]: false }));
      }
    },
    [
      afterAcceptMergeSuggestion,
      afterDismissMergeSuggestion,
      afterRejectMergeSuggestion,
      item.collection_id,
      item.id,
    ],
  );

  return (
    <div className="bg-card hover:bg-accent/70 flex flex-col gap-2 rounded-xl border px-4 py-3">
      <div className="flex flex-row items-center justify-between">
        <div
          className="hover:text-primary cursor-pointer font-serif text-base font-normal tracking-tight"
          onClick={() => onSelectNode(target.entity_name)}
        >
          {target.entity_name}
        </div>
        <div className="flex flex-row items-center gap-2">
          <Badge variant="outline" className={statusTone(item.status)}>
            {page_graph(STATUS_LABEL_KEYS[item.status])}
          </Badge>
          <div className="text-muted-foreground flex flex-row items-center gap-1 font-mono text-xs tabular-nums">
            <Sparkles className="size-3.5" />
            {Math.round(item.confidence_score * 100)}%
          </div>
          {canAct && (
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
          {canAct && (
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
          {canAct && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  size="icon"
                  variant="ghost"
                  className="cursor-pointer"
                  onClick={() => handleSuggestionAction('dismiss')}
                >
                  {loading?.dismiss ? (
                    <LoaderCircle className="animate-spin" />
                  ) : (
                    <Ban className="text-muted-foreground" />
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent>{page_graph('merge_dismiss')}</TooltipContent>
            </Tooltip>
          )}
        </div>
      </div>
      <div className="text-muted-foreground text-sm">
        {item.reason || item.merge_reason}
      </div>
      <div className="text-muted-foreground flex flex-wrap gap-2 text-xs">
        {target.entity_type && (
          <span>
            {page_graph('merge_target_type')}: {target.entity_type}
          </span>
        )}
        {observedTypes.length > 0 && (
          <span>
            {page_graph('merge_observed_types')}: {observedTypes.join(', ')}
          </span>
        )}
        {typeConflict && (
          <Badge variant="outline" className="border-amber-300 text-amber-700">
            {page_graph('merge_type_conflict')}
          </Badge>
        )}
        {affectedDocCount > 0 && (
          <span>
            {page_graph('merge_evidence_docs', {
              count: String(affectedDocCount),
            })}
          </span>
        )}
      </div>
      <div className="flex flex-wrap gap-1">
        {item.entity_ids.map((entity) => (
          <Badge
            key={entity}
            variant="outline"
            className="cursor-pointer"
            onClick={() => onSelectNode(entity)}
          >
            {resolveEntityLabel(item, entity)}
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
  onRun,
  running,
}: {
  dataSource: MergeSuggestionsResponse;
  open: boolean;
  onClose: () => void;
  onSelectNode: (id: string) => void;
  onRefresh: () => void;
  onRun: () => void;
  running: boolean;
}) => {
  const [activeStatus, setActiveStatus] =
    useState<MergeSuggestionStatus>('PENDING');
  const page_graph = useTranslations('page_graph');
  const suggestions = Array.isArray(dataSource.suggestions)
    ? dataSource.suggestions
    : [];
  const statusCounts = useMemo(() => {
    return suggestions.reduce(
      (counts, suggestion) => ({
        ...counts,
        [suggestion.status]: (counts[suggestion.status] || 0) + 1,
      }),
      {} as Partial<Record<MergeSuggestionStatus, number>>,
    );
  }, [suggestions]);
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
        <DrawerHeader className="flex flex-col gap-3 border-b">
          <div className="flex flex-row items-center justify-between gap-3">
            <DrawerTitle className="font-serif text-xl font-normal tracking-tight">
              {page_graph('merge_suggestions')}
            </DrawerTitle>
            <div className="flex items-center gap-2">
              <Button
                size="sm"
                variant="outline"
                className="cursor-pointer"
                onClick={onRefresh}
              >
                <RefreshCw className="size-4" />
                {page_graph('merge_refresh')}
              </Button>
              <Button
                size="sm"
                className="cursor-pointer"
                onClick={onRun}
                disabled={running}
              >
                {running ? (
                  <LoaderCircle className="size-4 animate-spin" />
                ) : (
                  <Sparkles className="size-4" />
                )}
                {page_graph('merge_run_scan')}
              </Button>
            </div>
          </div>
          <Tabs
            defaultValue={activeStatus}
            onValueChange={(v: string) =>
              setActiveStatus(v as MergeSuggestionStatus)
            }
          >
            <TabsList className="h-auto flex-wrap justify-start">
              {STATUS_FILTERS.map((status) => (
                <TabsTrigger key={status} value={status}>
                  {page_graph(STATUS_LABEL_KEYS[status])}
                  {statusCounts[status] ? ` ${statusCounts[status]}` : ''}
                </TabsTrigger>
              ))}
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
                  afterDismissMergeSuggestion={onRefresh}
                />
              );
            })
          )}
        </div>
      </DrawerContent>
    </Drawer>
  );
};
