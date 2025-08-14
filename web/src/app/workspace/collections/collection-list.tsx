'use client';

import { CollectionView, CollectionViewStatusEnum } from '@/api';
import { FormatDate } from '@/components/format-date';
import {
  Card,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { cn } from '@/lib/utils';
import _ from 'lodash';
import { Calendar } from 'lucide-react';
import Link from 'next/link';

export const CollectionList = ({
  collections,
}: {
  collections: CollectionView[];
}) => {
  if (collections.length === 0) {
    return (
      <div className="text-muted-foreground my-40 text-center">
        No collections found
      </div>
    );
  }

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {collections.map((collection) => {
        const badgeColor: {
          [key in CollectionViewStatusEnum]: string;
        } = {
          ACTIVE: 'bg-green-500',
          INACTIVE: 'bg-red-500',
          DELETED: 'bg-gray-500',
        };
        return (
          <Link
            key={collection.id}
            href={
              collection.subscription_id
                ? `/marketplace/collections/${collection.id}`
                : `/workspace/collections/${collection.id}/general`
            }
          >
            <Card className="cursor-pointer rounded-md hover:mask-alpha">
              <CardHeader>
                <CardTitle className="truncate">{collection.title}</CardTitle>
                <CardDescription className="h-4 truncate">
                  {collection.description}
                </CardDescription>
              </CardHeader>
              <CardFooter className="justify-between">
                <div className="text-muted-foreground text-sm">
                  {collection.created && (
                    <div className="flex items-center gap-2">
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
              </CardFooter>
            </Card>
          </Link>
        );
      })}
    </div>
  );
};
