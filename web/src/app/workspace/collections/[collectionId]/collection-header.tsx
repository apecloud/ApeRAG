'use client';
import { FormatDate } from '@/components/format-date';
import { PageContent } from '@/components/page-container';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardAction,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import type { CollectionStatus } from '@/features/collection/types';
import { cn } from '@/lib/utils';
import _ from 'lodash';

import { CollectionExport } from '@/components/collections/export-dialog';
import { useCollectionContext } from '@/components/providers/collection-provider';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  publishCollectionSharing,
  unpublishCollectionSharing,
} from '@/features/collection/client-api';
import {
  Calendar,
  Database,
  Download,
  EllipsisVertical,
  Files,
  FolderSearch,
  Settings,
  Trash,
  VectorSquare,
} from 'lucide-react';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useCallback, useMemo } from 'react';
import { toast } from 'sonner';
import { showRetrievalTestModule } from '../feature-visibility';
import { CollectionDelete } from './collection-delete';

export const CollectionHeader = ({ className }: { className?: string }) => {
  const statusClassName: Record<CollectionStatus, string> = {
    ACTIVE: 'bg-accent-soft text-accent-ink border-accent-soft',
    INACTIVE: 'bg-secondary text-muted-foreground border-transparent',
    DELETED: 'bg-secondary text-muted-foreground border-transparent',
  };
  const { collection, share, loadShare } = useCollectionContext();
  const pathname = usePathname();
  const page_collections = useTranslations('page_collections');
  const page_documents = useTranslations('page_documents');
  const page_graph = useTranslations('page_graph');
  const page_search = useTranslations('page_search');

  const urls = useMemo(() => {
    return {
      documents: `/workspace/collections/${collection.id}/documents`,
      evaluations: `/workspace/collections/${collection.id}/evaluations`,
      search: `/workspace/collections/${collection.id}/search`,
      graph: `/workspace/collections/${collection.id}/graph-hybrid`,
      settings: `/workspace/collections/${collection.id}/settings`,
    };
  }, [collection.id]);

  const shareCollection = useCallback(
    async (checked: boolean) => {
      if (!collection?.id) {
        return;
      }
      if (checked) {
        await publishCollectionSharing(collection.id);
        toast.success(page_collections('published_success'));
      } else {
        await unpublishCollectionSharing(collection.id);
        toast.success(page_collections('unpublished_success'));
      }
      await loadShare();
    },
    [collection?.id, loadShare, page_collections],
  );

  const navItems = useMemo(
    () =>
      [
        {
          href: urls.documents,
          active: Boolean(pathname.match(urls.documents)),
          icon: Files,
          label: page_documents('metadata.title'),
        },
        collection.config?.enable_knowledge_graph
          ? {
              href: urls.graph,
              active: Boolean(pathname.match(urls.graph)),
              icon: VectorSquare,
              label: page_graph('metadata.title'),
            }
          : null,
        showRetrievalTestModule
          ? {
              href: urls.search,
              active: Boolean(pathname.match(urls.search)),
              icon: FolderSearch,
              label: page_search('metadata.title'),
            }
          : null,
        {
          href: urls.settings,
          active: Boolean(pathname.match(urls.settings)),
          icon: Settings,
          label: page_collections('settings'),
        },
      ].filter(Boolean),
    [
      collection.config?.enable_knowledge_graph,
      page_collections,
      page_documents,
      page_graph,
      page_search,
      pathname,
      urls.documents,
      urls.graph,
      urls.search,
      urls.settings,
    ],
  );

  return (
    <PageContent className={cn('flex flex-col gap-4 pb-0', className)}>
      <Card className="border-border/70 gap-0 overflow-hidden rounded-xl py-0 shadow-sm">
        <CardHeader className="gap-4 p-5 lg:grid-cols-[1fr_auto]">
          <div className="flex min-w-0 gap-4">
            <div className="bg-accent-soft text-accent-ink flex size-11 shrink-0 items-center justify-center rounded-xl">
              <Database className="size-5" />
            </div>
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <CardTitle className="truncate text-2xl leading-8 font-medium">
                  {collection.title}
                </CardTitle>
                {collection.status && (
                  <Badge
                    className={cn(
                      'rounded-sm border',
                      statusClassName[collection.status as CollectionStatus],
                    )}
                    variant="outline"
                  >
                    {_.upperFirst(_.lowerCase(collection.status))}
                  </Badge>
                )}
              </div>
              {collection.description && (
                <CardDescription className="mt-2 max-w-3xl leading-6">
                  {_.truncate(collection.description, {
                    length: 220,
                  })}
                </CardDescription>
              )}
              {collection.created && (
                <div className="text-muted-foreground mt-3 flex items-center gap-1.5 text-xs">
                  <Calendar className="size-3.5" />
                  <FormatDate datetime={new Date(collection.created)} />
                </div>
              )}
            </div>
          </div>

          <CardAction className="flex flex-row items-center gap-2">
            {share && (
              <Badge
                className={cn(
                  'rounded-sm',
                  share.is_published
                    ? 'bg-accent-soft text-accent-ink border-accent-soft border'
                    : '',
                )}
                variant={share.is_published ? 'outline' : 'secondary'}
              >
                {share.is_published
                  ? page_collections('public')
                  : page_collections('private')}
              </Badge>
            )}

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button size="icon" variant="ghost">
                  <EllipsisVertical />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-60">
                {share && (
                  <>
                    {share.is_published ? (
                      <DropdownMenuItem
                        className="flex-col items-start gap-1"
                        onClick={() => shareCollection(false)}
                      >
                        <div>{page_collections('unpublish_collection')}</div>
                        <div className="text-muted-foreground text-xs">
                          {page_collections('unpublish_collection_description')}
                        </div>
                      </DropdownMenuItem>
                    ) : (
                      <DropdownMenuItem
                        className="flex-col items-start gap-1"
                        onClick={() => shareCollection(true)}
                      >
                        <div>{page_collections('publish_collection')}</div>
                        <div className="text-muted-foreground text-xs">
                          {page_collections('publish_collection_description')}
                        </div>
                      </DropdownMenuItem>
                    )}
                    <DropdownMenuSeparator />
                  </>
                )}

                {share && (
                  <>
                    <CollectionExport collectionId={collection.id ?? ''}>
                      <DropdownMenuItem>
                        <Download /> {page_collections('export_knowledge_base')}
                      </DropdownMenuItem>
                    </CollectionExport>
                    <DropdownMenuSeparator />
                  </>
                )}

                <CollectionDelete>
                  <DropdownMenuItem variant="destructive">
                    <Trash /> {page_collections('delete_collection')}
                  </DropdownMenuItem>
                </CollectionDelete>
              </DropdownMenuContent>
            </DropdownMenu>
          </CardAction>
        </CardHeader>
        <Separator />
        <div className="bg-muted/50 flex gap-1 overflow-x-auto px-3 py-2">
          {navItems.map((item) => {
            if (!item) return null;
            const Icon = item.icon;
            return (
              <Button
                key={item.href}
                asChild
                data-active={item.active}
                className="data-[active=true]:bg-card data-[active=true]:text-foreground text-muted-foreground h-9 shrink-0 rounded-lg px-3 data-[active=true]:shadow-xs"
                variant="ghost"
              >
                <Link href={item.href}>
                  <Icon className="size-4" />
                  <span>{item.label}</span>
                </Link>
              </Button>
            );
          })}
        </div>
      </Card>
    </PageContent>
  );
};
