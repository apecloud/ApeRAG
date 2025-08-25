'use client';

import { SharedCollection } from '@/api';
import { PageContent } from '@/components/page-container';
import { useAppContext } from '@/components/providers/app-provider';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardAction,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { apiClient } from '@/lib/api/client';
import { Calendar, Files, User, VectorSquare } from 'lucide-react';
import { useFormatter } from 'next-intl';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { useCallback, useMemo } from 'react';
import { FaStar } from 'react-icons/fa6';

export const CollectionHeader = ({
  collection,
}: {
  collection: SharedCollection;
}) => {
  const router = useRouter();
  const format = useFormatter();
  const pathname = usePathname();

  const { user } = useAppContext();

  const isOwner = useMemo(
    () => collection.owner_user_id === user?.id,
    [collection.owner_user_id, user?.id],
  );
  const isSubscriber = useMemo(
    () => collection.subscription_id !== null,
    [collection.subscription_id],
  );

  const handleSubscribe = useCallback(async () => {
    if (isSubscriber) {
      await apiClient.defaultApi.marketplaceCollectionsCollectionIdSubscribeDelete(
        {
          collectionId: collection.id,
        },
      );
    } else {
      await apiClient.defaultApi.marketplaceCollectionsCollectionIdSubscribePost(
        {
          collectionId: collection.id,
        },
      );
    }
    router.refresh();
  }, [collection.id, isSubscriber, router]);

  return (
    <PageContent className="flex flex-col gap-4 pb-0">
      <Card className="gap-0 p-0">
        <CardHeader className="p-4">
          <CardTitle className="text-2xl">{collection.title}</CardTitle>
          <CardDescription>
            {collection.description || 'No description available'}
          </CardDescription>
          <CardAction className="text-muted-foreground flex flex-row items-center gap-4 text-xs">
            {isOwner ? (
              <Badge>Mine</Badge>
            ) : (
              <div className="flex flex-row items-center gap-1">
                <User className="size-4" />
                <div className="max-w-60 truncate">
                  {collection.owner_username}
                </div>
              </div>
            )}

            {collection.gmt_subscribed && (
              <div className="text-muted-foreground flex items-center gap-1 text-xs">
                <Calendar className="size-3" />
                {format.dateTime(new Date(collection.gmt_subscribed), 'medium')}
              </div>
            )}
            <Button
              variant="ghost"
              size="icon"
              hidden={isOwner}
              onClick={handleSubscribe}
              data-subscriber={isSubscriber}
              className="text-muted-foreground cursor-pointer data-[subscriber=true]:text-orange-500"
            >
              <FaStar />
            </Button>
            {/* {isOwner && (
              <Button
                className="cursor-pointer"
                variant="secondary"
                size="icon"
                onClick={(e) => {
                  e.preventDefault();
                  router.push(
                    `/workspace/collections/${collection.id}/documents`,
                  );
                }}
              >
                <Settings />
              </Button>
            )} */}
          </CardAction>
        </CardHeader>
        <Separator />
        <div className="bg-accent/50 flex flex-row gap-2 rounded-b-xl px-4">
          <Button
            asChild
            data-active={Boolean(
              pathname.match(
                `/workspace/market/collections/${collection.id}/documents`,
              ),
            )}
            className="data-[active=true]:border-b-primary h-10 rounded-none border-y-2 border-y-transparent px-1 has-[>svg]:px-2"
            variant="ghost"
          >
            <Link
              href={`/workspace/market/collections/${collection.id}/documents`}
            >
              <Files />
              <span className="hidden lg:inline">Documents</span>
            </Link>
          </Button>

          {/* {collection.config?.enable_knowledge_graph && ( */}
          <Button
            asChild
            data-active={Boolean(
              pathname.match(
                `/workspace/market/collections/${collection.id}/graph`,
              ),
            )}
            className="data-[active=true]:border-b-primary h-10 rounded-none border-y-2 border-y-transparent px-1 has-[>svg]:px-2"
            variant="ghost"
          >
            <Link href={`/workspace/market/collections/${collection.id}/graph`}>
              <VectorSquare />
              <span className="hidden lg:inline">Knowledge Graph</span>
            </Link>
          </Button>
          {/* )} */}
        </div>
      </Card>
    </PageContent>
  );
};
