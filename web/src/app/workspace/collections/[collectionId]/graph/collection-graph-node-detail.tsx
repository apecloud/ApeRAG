'use client';

import { Markdown } from '@/components/markdown';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Drawer,
  DrawerContent,
  DrawerHeader,
  DrawerTitle,
} from '@/components/ui/drawer';
import { Separator } from '@/components/ui/separator';
import type { GraphEdge, GraphNode } from '@/features/knowledge-graph/types';
import { ENTITY_PALETTE, entityTypeToPaletteKey } from '@/lib/design-tokens';
import { cn } from '@/lib/utils';
import { ArrowRight, CircleDot, GitBranch, Hash } from 'lucide-react';
import { useTranslations } from 'next-intl';

const endpointId = (endpoint: unknown) => {
  if (typeof endpoint === 'object' && endpoint !== null && 'id' in endpoint) {
    return String(endpoint.id);
  }
  return String(endpoint || '');
};

const endpointNode = (endpoint: unknown) => {
  if (typeof endpoint === 'object' && endpoint !== null && 'id' in endpoint) {
    return endpoint as GraphNode;
  }
  return undefined;
};

const entityTitleKey = (entityType: string) => `entity_${entityType}`;

const EntityTypeBadge = ({ node }: { node?: GraphNode }) => {
  const page_graph = useTranslations('page_graph');
  const entityType = node?.properties.entity_type || 'UNKNOWN';
  const entityColor = ENTITY_PALETTE[entityTypeToPaletteKey(entityType)];
  const entityLabel = node
    ? // @ts-expect-error dynamic i18n key
      page_graph(entityTitleKey(entityType))
    : '';

  return (
    <div className="flex items-center gap-2">
      <span
        className="size-2 shrink-0 rounded-full"
        style={{ backgroundColor: entityColor }}
      />
      <span className="text-muted-foreground font-mono text-[10px] tracking-wider uppercase">
        {entityLabel}
      </span>
    </div>
  );
};

const InfoTile = ({
  label,
  value,
}: {
  label: string;
  value: string | number;
}) => (
  <div className="bg-muted/45 rounded-lg px-3 py-2">
    <div className="text-muted-foreground text-[11px]">{label}</div>
    <div className="mt-1 text-sm font-medium tabular-nums">{value}</div>
  </div>
);

const parseKeywords = (keywords?: string | null) =>
  (keywords || '')
    .split(/[,，、]/)
    .map((keyword) => keyword.trim())
    .filter(Boolean);

export const CollectionGraphNodeDetail = ({
  open,
  node,
  relatedEdges = [],
  relatedNodes = [],
  onClose,
  onSelectEdge,
  onSelectNode,
}: {
  open: boolean;
  node?: GraphNode;
  relatedEdges: GraphEdge[];
  relatedNodes: GraphNode[];
  onClose: () => void;
  onSelectEdge: (edge: GraphEdge) => void;
  onSelectNode: (node: GraphNode) => void;
}) => {
  const page_graph = useTranslations('page_graph');

  return (
    <Drawer
      direction="right"
      open={open}
      onOpenChange={onClose}
      handleOnly={true}
    >
      <DrawerContent className="flex sm:min-w-sm md:min-w-md lg:min-w-lg">
        <DrawerHeader className="gap-2 border-b">
          <EntityTypeBadge node={node} />
          <DrawerTitle className="font-serif text-xl leading-tight font-normal tracking-tight">
            {node?.id}
          </DrawerTitle>
        </DrawerHeader>
        <div className="flex-1 space-y-5 overflow-auto p-4 text-sm leading-relaxed select-text">
          <section className="space-y-2">
            <div className="text-muted-foreground text-xs font-medium">
              {page_graph('entity_description')}
            </div>
            {node?.properties.description ? (
              <Markdown>{node.properties.description}</Markdown>
            ) : (
              <p className="text-muted-foreground italic">
                {page_graph('no_description')}
              </p>
            )}
          </section>

          <section className="grid grid-cols-3 gap-2">
            <InfoTile
              label={page_graph('neighbors')}
              value={relatedNodes.length}
            />
            <InfoTile
              label={page_graph('relations')}
              value={relatedEdges.length}
            />
            <InfoTile
              label={page_graph('source_chunks')}
              value={node?.properties.source_chunk_count || 0}
            />
          </section>

          {relatedNodes.length > 0 && (
            <section className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <div className="text-muted-foreground text-xs font-medium">
                  {page_graph('neighbor_entities')}
                </div>
                <Badge variant="secondary">{relatedNodes.length}</Badge>
              </div>
              <div className="space-y-1.5">
                {relatedNodes.slice(0, 12).map((relatedNode) => (
                  <button
                    key={relatedNode.id}
                    className="hover:bg-muted/70 flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left transition-colors"
                    onClick={() => onSelectNode(relatedNode)}
                  >
                    <EntityTypeBadge node={relatedNode} />
                    <span className="min-w-0 flex-1 truncate font-medium">
                      {relatedNode.id}
                    </span>
                    <ArrowRight className="text-muted-foreground size-3.5" />
                  </button>
                ))}
              </div>
            </section>
          )}

          {relatedEdges.length > 0 && (
            <section className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <div className="text-muted-foreground text-xs font-medium">
                  {page_graph('related_relations')}
                </div>
                <Badge variant="secondary">{relatedEdges.length}</Badge>
              </div>
              <div className="space-y-2">
                {relatedEdges.slice(0, 16).map((edge) => {
                  const sourceId = endpointId(edge.source);
                  const targetId = endpointId(edge.target);
                  const otherId = sourceId === node?.id ? targetId : sourceId;
                  return (
                    <button
                      key={edge.id}
                      className="hover:bg-muted/70 w-full rounded-lg border p-2 text-left transition-colors"
                      onClick={() => onSelectEdge(edge)}
                    >
                      <div className="flex items-center gap-2 text-xs">
                        <GitBranch className="text-muted-foreground size-3.5" />
                        <span className="min-w-0 flex-1 truncate font-medium">
                          {otherId}
                        </span>
                        {edge.properties.weight != null && (
                          <span className="text-muted-foreground tabular-nums">
                            {page_graph('weight_short', {
                              count: String(edge.properties.weight),
                            })}
                          </span>
                        )}
                      </div>
                      {edge.properties.description && (
                        <div className="text-muted-foreground mt-1 line-clamp-2 text-xs">
                          {edge.properties.description}
                        </div>
                      )}
                    </button>
                  );
                })}
              </div>
            </section>
          )}
        </div>
      </DrawerContent>
    </Drawer>
  );
};

export const CollectionGraphEdgeDetail = ({
  open,
  edge,
  onClose,
  onSelectNode,
}: {
  open: boolean;
  edge?: GraphEdge;
  onClose: () => void;
  onSelectNode: (node: GraphNode) => void;
}) => {
  const page_graph = useTranslations('page_graph');
  const sourceNode = endpointNode(edge?.source);
  const targetNode = endpointNode(edge?.target);
  const sourceId = endpointId(edge?.source);
  const targetId = endpointId(edge?.target);
  const keywords = parseKeywords(edge?.properties.keywords);

  return (
    <Drawer
      direction="right"
      open={open}
      onOpenChange={onClose}
      handleOnly={true}
    >
      <DrawerContent className="flex sm:min-w-sm md:min-w-md lg:min-w-lg">
        <DrawerHeader className="gap-2 border-b">
          <div className="text-muted-foreground flex items-center gap-2 font-mono text-[10px] tracking-wider uppercase">
            <GitBranch className="size-3.5" />
            {page_graph('relation_detail')}
          </div>
          <DrawerTitle className="font-serif text-xl leading-tight font-normal tracking-tight">
            {sourceId} <ArrowRight className="mx-1 inline size-4" /> {targetId}
          </DrawerTitle>
        </DrawerHeader>

        <div className="flex-1 space-y-5 overflow-auto p-4 text-sm leading-relaxed select-text">
          <section className="grid grid-cols-3 gap-2">
            <InfoTile
              label={page_graph('weight')}
              value={edge?.properties.weight ?? '-'}
            />
            <InfoTile
              label={page_graph('source_chunks')}
              value={edge?.properties.source_chunk_count || 0}
            />
            <InfoTile
              label={page_graph('relation_type')}
              value={edge?.type || '-'}
            />
          </section>

          <section className="space-y-2">
            <div className="text-muted-foreground text-xs font-medium">
              {page_graph('relation_description')}
            </div>
            {edge?.properties.description ? (
              <Markdown>{edge.properties.description}</Markdown>
            ) : (
              <p className="text-muted-foreground italic">
                {page_graph('no_description')}
              </p>
            )}
          </section>

          {keywords.length > 0 && (
            <section className="space-y-2">
              <div className="text-muted-foreground text-xs font-medium">
                {page_graph('keywords')}
              </div>
              <div className="flex flex-wrap gap-1.5">
                {keywords.map((keyword) => (
                  <Badge key={keyword} variant="secondary">
                    <Hash className="size-3" />
                    {keyword}
                  </Badge>
                ))}
              </div>
            </section>
          )}

          <Separator />

          <section className="space-y-2">
            <div className="text-muted-foreground text-xs font-medium">
              {page_graph('relation_endpoints')}
            </div>
            <EndpointButton
              node={sourceNode}
              fallbackId={sourceId}
              label={page_graph('source_entity')}
              onSelectNode={onSelectNode}
            />
            <EndpointButton
              node={targetNode}
              fallbackId={targetId}
              label={page_graph('target_entity')}
              onSelectNode={onSelectNode}
            />
          </section>
        </div>
      </DrawerContent>
    </Drawer>
  );
};

const EndpointButton = ({
  node,
  fallbackId,
  label,
  onSelectNode,
}: {
  node?: GraphNode;
  fallbackId: string;
  label: string;
  onSelectNode: (node: GraphNode) => void;
}) => (
  <Button
    variant="outline"
    className={cn('h-auto w-full justify-start gap-2 px-3 py-2')}
    disabled={!node}
    onClick={() => node && onSelectNode(node)}
  >
    <CircleDot className="text-muted-foreground size-3.5" />
    <span className="text-muted-foreground text-xs">{label}</span>
    <span className="min-w-0 flex-1 truncate text-left font-medium">
      {fallbackId}
    </span>
  </Button>
);
