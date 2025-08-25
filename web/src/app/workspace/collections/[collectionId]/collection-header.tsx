'use client';
import { CollectionViewStatusEnum } from '@/api';
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
import { cn } from '@/lib/utils';
import _ from 'lodash';

import { useCollectionContext } from '@/components/providers/collection-provider';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { apiClient } from '@/lib/api/client';
import {
  Calendar,
  EllipsisVertical,
  Files,
  FolderSearch,
  Settings,
  Trash,
  VectorSquare,
} from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useCallback, useMemo } from 'react';
import { toast } from 'sonner';
import { CollectionDelete } from './collection-delete';

export const CollectionHeader = () => {
  const badgeColor: {
    [key in CollectionViewStatusEnum]: string;
  } = {
    ACTIVE: 'bg-green-700',
    INACTIVE: 'bg-red-500',
    DELETED: 'bg-gray-500',
  };
  const { collection, share, loadShare } = useCollectionContext();
  const pathname = usePathname();

  const urls = useMemo(() => {
    return {
      documents: `/workspace/collections/${collection.id}/documents`,
      search: `/workspace/collections/${collection.id}/search`,
      graph: `/workspace/collections/${collection.id}/graph`,
      settings: `/workspace/collections/${collection.id}/settings`,
    };
  }, [collection.id]);

  const shareCollection = useCallback(
    async (checked: boolean) => {
      if (!collection?.id) {
        return;
      }
      if (checked) {
        await apiClient.defaultApi.collectionsCollectionIdSharingPost({
          collectionId: collection?.id,
        });
        toast.success('Collection published successfully');
      } else {
        await apiClient.defaultApi.collectionsCollectionIdSharingDelete({
          collectionId: collection?.id,
        });
        toast.success('Collection unpublished successfully');
      }
      await loadShare();
    },
    [collection?.id, loadShare],
  );

  return (
    <PageContent className="flex flex-col gap-4 pb-0">
      <Card className="gap-0 p-0">
        <CardHeader className="p-4">
          <CardTitle className="text-2xl">{collection.title}</CardTitle>
          <CardDescription className="flex flex-row items-center gap-6">
            <div>
              {collection.created && (
                <div className="text-muted-foreground flex items-center gap-1 text-sm">
                  <Calendar className="size-3" />
                  <FormatDate datetime={new Date(collection.created)} />
                </div>
              )}
            </div>
            <div className="flex items-center gap-2">
              <div
                className={cn(
                  'size-2 rounded-2xl',
                  collection.status
                    ? badgeColor[collection.status]
                    : 'bg-gray-500',
                )}
              />
              <div className="text-muted-foreground text-sm">
                {_.upperFirst(_.lowerCase(collection.status))}
              </div>
            </div>
          </CardDescription>
          <CardAction className="flex flex-row items-center gap-4">
            {share && (
              <Badge variant={share.is_published ? 'default' : 'secondary'}>
                {share.is_published ? 'Public' : 'Private'}
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
                        <div>Unpublish from Marketplace</div>
                        <div className="text-muted-foreground text-xs">
                          Remove the collection from the marketplace, making it
                          private.
                        </div>
                      </DropdownMenuItem>
                    ) : (
                      <DropdownMenuItem
                        className="flex-col items-start gap-1"
                        onClick={() => shareCollection(true)}
                      >
                        <div>Publish to Marketplace</div>
                        <div className="text-muted-foreground text-xs">
                          Allow users to discover and subscribe to your
                          collection.
                        </div>
                      </DropdownMenuItem>
                    )}
                    <DropdownMenuSeparator />
                  </>
                )}

                <CollectionDelete>
                  <DropdownMenuItem variant="destructive">
                    <Trash /> Delete Collection
                  </DropdownMenuItem>
                </CollectionDelete>
              </DropdownMenuContent>
            </DropdownMenu>
          </CardAction>
        </CardHeader>
        <Separator />
        <div className="bg-accent/50 flex flex-row gap-4 rounded-b-xl px-4">
          <Button
            asChild
            data-active={Boolean(pathname.match(urls.documents))}
            className="hover:border-b-primary data-[active=true]:border-b-primary h-10 rounded-none border-y-2 border-y-transparent px-1 has-[>svg]:px-2"
            variant="ghost"
          >
            <Link href={urls.documents}>
              <Files />
              <span className="hidden sm:inline">Documents</span>
            </Link>
          </Button>

          {collection.config?.enable_knowledge_graph && (
            <Button
              asChild
              data-active={Boolean(pathname.match(urls.graph))}
              className="hover:border-b-primary data-[active=true]:border-b-primary h-10 rounded-none border-y-2 border-y-transparent px-1 has-[>svg]:px-2"
              variant="ghost"
            >
              <Link href={urls.graph}>
                <VectorSquare />
                <span className="hidden sm:inline">Knowledge Graph</span>
              </Link>
            </Button>
          )}

          <Button
            asChild
            data-active={Boolean(pathname.match(urls.search))}
            className="hover:border-b-primary data-[active=true]:border-b-primary h-10 rounded-none border-y-2 border-y-transparent px-1 has-[>svg]:px-2"
            variant="ghost"
          >
            <Link href={urls.search}>
              <FolderSearch />
              <span className="hidden sm:inline">Search Effect</span>
            </Link>
          </Button>

          <Button
            asChild
            data-active={Boolean(pathname.match(urls.settings))}
            className="hover:border-b-primary data-[active=true]:border-b-primary h-10 rounded-none border-y-2 border-y-transparent px-1 has-[>svg]:px-2"
            variant="ghost"
          >
            <Link href={urls.settings}>
              <Settings /> <span className="hidden sm:inline">Settings</span>
            </Link>
          </Button>
        </div>
      </Card>
    </PageContent>
  );
};
