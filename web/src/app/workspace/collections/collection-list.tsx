'use client';

import { CollectionView, CollectionViewStatusEnum } from '@/api';
import { FormatDate } from '@/components/format-date';
import { Badge } from '@/components/ui/badge';
import {
  Card,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
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
          ACTIVE: 'text-green-500',
          INACTIVE: 'text-red-500',
          DELETED: '',
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
                <CardDescription className="truncate h-4">
                  {collection.description}
                </CardDescription>
              </CardHeader>
              <CardFooter className="justify-between">
                <div>
                  {collection.created && (
                    <div className="text-muted-foreground flex items-center gap-1 text-sm">
                      <Calendar className="size-3" />
                      <FormatDate datetime={new Date(collection.created)} />
                    </div>
                  )}
                </div>
                <div>
                  {collection.status && (
                    <Badge
                      variant="secondary"
                      className={badgeColor[collection.status]}
                    >
                      {_.upperFirst(_.lowerCase(collection.status))}
                    </Badge>
                  )}
                </div>
              </CardFooter>
            </Card>
          </Link>
        );
      })}
    </div>
  );
};
