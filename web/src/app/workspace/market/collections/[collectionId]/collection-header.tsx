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
import { apiClient } from '@/lib/api/client';
import { Calendar, Settings, Star, User } from 'lucide-react';
import { useFormatter } from 'next-intl';
import { useRouter } from 'next/navigation';
import { useCallback, useMemo } from 'react';

export const CollectionHeader = ({
  collection,
}: {
  collection: SharedCollection;
}) => {
  const router = useRouter();
  const format = useFormatter();

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
          <CardTitle>{collection.title}</CardTitle>
          <CardDescription className="mb-2 flex flex-row items-center gap-6">
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
          </CardDescription>
          <CardDescription>
            {collection.description || 'No description available'}
          </CardDescription>
          <CardAction className="flex flex-row items-center gap-2">
            {collection.gmt_subscribed && (
              <div className="text-muted-foreground flex items-center gap-1 text-xs">
                <Calendar className="size-3" />
                {format.dateTime(new Date(collection.gmt_subscribed), 'medium')}
              </div>
            )}
            <Button
              variant={isSubscriber ? 'default' : 'secondary'}
              size="icon"
              hidden={isOwner}
              onClick={handleSubscribe}
              className="cursor-pointer"
            >
              <Star />
            </Button>
            {isOwner && (
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
            )}
          </CardAction>
        </CardHeader>
      </Card>
    </PageContent>
  );
};
