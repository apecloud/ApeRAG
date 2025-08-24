'use client';

import { CollectionView, CollectionViewStatusEnum } from '@/api';
import { FormatDate } from '@/components/format-date';
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
import { cn } from '@/lib/utils';
import _ from 'lodash';
import { Calendar, Plus } from 'lucide-react';
import Link from 'next/link';
import { useState } from 'react';
import { FaStar } from 'react-icons/fa6';

export const CollectionList = ({
  collections,
}: {
  collections: CollectionView[];
}) => {
  const [searchValue, setSearchValue] = useState<string>('');

  if (collections.length === 0) {
    return (
      <div className="text-muted-foreground my-40 text-center">
        No collections found
      </div>
    );
  }

  return (
    <>
      <div className="mb-4 flex flex-row items-center">
        <div>
          <Input
            placeholder="Search"
            value={searchValue}
            onChange={(e) => setSearchValue(e.currentTarget.value)}
          />
        </div>
        <div className="ml-auto flex items-center gap-2">
          <Button asChild>
            <Link href="/workspace/collections/new">
              <Plus /> Add collection
            </Link>
          </Button>
        </div>
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
            const badgeColor: {
              [key in CollectionViewStatusEnum]: string;
            } = {
              ACTIVE: 'bg-green-700',
              INACTIVE: 'bg-red-500',
              DELETED: 'bg-gray-500',
            };
            return (
              <Link
                key={collection.id}
                href={
                  collection.subscription_id
                    ? `/workspace/market/collections/${collection.id}/documents`
                    : `/workspace/collections/${collection.id}/documents`
                }
                target={collection.subscription_id ? '_blank' : '_self'}
              >
                <Card className="hover:bg-accent/70 h-34 cursor-pointer rounded-md">
                  <CardHeader className="px-4">
                    <CardTitle className="h-5 truncate">
                      {collection.title}
                    </CardTitle>
                    <CardDescription className="h-5 truncate">
                      {collection.description || 'No description available'}
                    </CardDescription>
                    <CardAction className="flex flex-row items-center gap-4">
                      <Badge
                        variant="secondary"
                        data-published={collection.is_published}
                        className="data-[published=true]:bg-primary data-[published=true]:text-primary-foreground"
                      >
                        {collection.is_published ? 'Public' : 'Private'}
                      </Badge>
                      {collection.subscription_id && (
                        <div className="size-4 text-amber-500">
                          <FaStar />
                        </div>
                      )}
                    </CardAction>
                  </CardHeader>
                  <CardFooter className="justify-between px-4 text-xs">
                    <div className="text-muted-foreground">
                      {collection.created && (
                        <div className="flex items-center gap-2">
                          <Calendar className="size-3" />
                          <FormatDate datetime={new Date(collection.created)} />
                        </div>
                      )}
                    </div>
                    <div className="flex items-center gap-1">
                      <div
                        className={cn(
                          'size-2 rounded-2xl',
                          collection.status
                            ? badgeColor[collection.status]
                            : 'bg-gray-500',
                        )}
                      />
                      <div className="text-muted-foreground">
                        {_.upperFirst(_.lowerCase(collection.status))}
                      </div>
                    </div>
                  </CardFooter>
                </Card>
              </Link>
            );
          })}
      </div>
    </>
  );
};
