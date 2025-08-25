'use client';

import { SharedCollection } from '@/api';
import { useAppContext } from '@/components/providers/app-provider';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardAction,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { apiClient } from '@/lib/api/client';
import { User } from 'lucide-react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useCallback, useState } from 'react';
import { FaStar } from 'react-icons/fa6';

export const CollectionList = ({
  collections,
}: {
  collections: SharedCollection[];
}) => {
  const { user } = useAppContext();
  const [searchValue, setSearchValue] = useState<string>('');
  const router = useRouter();

  const handleSubscribe = useCallback(
    async (collection: SharedCollection) => {
      const isSubscriber = collection.subscription_id !== null;
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
    },
    [router],
  );

  if (collections.length === 0) {
    return (
      <div className="text-muted-foreground my-40 text-center">
        No collections found
      </div>
    );
  }

  return (
    <>
      <div className="mb-4">
        <Input
          placeholder="Search"
          value={searchValue}
          onChange={(e) => setSearchValue(e.currentTarget.value)}
          className="max-w-md"
        />
      </div>
      <div className="sm:grid-col-1 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {collections
          .filter((collection) => {
            if (searchValue === '') return true;
            return (
              collection.title?.match(new RegExp(searchValue)) ||
              collection.description?.match(new RegExp(searchValue))
            );
          })
          .map((collection) => {
            const isOwner = collection.owner_user_id === user?.id;
            const isSubscriber = collection.subscription_id !== null;
            return (
              <Link
                key={collection.id}
                href={`/workspace/market/collections/${collection.id}/documents`}
              >
                <Card className="hover:bg-accent/70 h-34 cursor-pointer rounded-md">
                  <CardHeader className="px-4">
                    <CardTitle className="h-5 truncate">
                      {collection.title}
                    </CardTitle>
                    <CardDescription className="h-5 truncate">
                      {collection.description || 'No description available'}
                    </CardDescription>
                    <CardAction className="flex flex-row gap-2">
                      <Button
                        variant="ghost"
                        hidden={isOwner}
                        onClick={(e) => {
                          handleSubscribe(collection);
                          e.preventDefault();
                        }}
                        data-subscriber={isSubscriber}
                        className="text-muted-foreground size-8 cursor-pointer data-[subscriber=true]:text-orange-500"
                      >
                        <FaStar />
                      </Button>
                      {/* {isOwner && (
                        <Button
                          className="size-8 cursor-pointer"
                          variant="secondary"
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
                  <CardFooter className="text-muted-foreground justify-between px-4 text-sm">
                    {isOwner ? (
                      <Badge>Mine</Badge>
                    ) : (
                      <div className="flex flex-row items-center gap-1">
                        <User className="size-4" />
                        <div className="max-w-20 truncate">
                          {collection.owner_username || '--'}
                        </div>
                      </div>
                    )}
                  </CardFooter>
                </Card>
              </Link>
            );
          })}
      </div>
    </>
  );
};
