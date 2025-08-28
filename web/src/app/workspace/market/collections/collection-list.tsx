'use client';

import { SharedCollection } from '@/api';
import { useAppContext } from '@/components/providers/app-provider';
import { Badge } from '@/components/ui/badge';
import {
  Card,
  CardAction,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { User } from 'lucide-react';
import Link from 'next/link';
import { useState } from 'react';

export const CollectionList = ({
  collections,
}: {
  collections: SharedCollection[];
}) => {
  const { user } = useAppContext();
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
            return (
              <Link
                key={collection.id}
                href={`/workspace/market/collections/${collection.id}/documents`}
              >
                <Card className="hover:bg-accent/30 cursor-pointer gap-2 rounded-md">
                  <CardHeader className="px-4">
                    <CardTitle className="h-5 truncate">
                      {collection.title}
                    </CardTitle>

                    <CardAction className="flex flex-row gap-2">
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
                  <CardDescription className="mb-4 truncate px-4">
                    {collection.description || 'No description available'}
                  </CardDescription>
                  <CardFooter className="text-muted-foreground justify-between px-4 text-sm">
                    {isOwner ? (
                      <Badge>Mine</Badge>
                    ) : (
                      <div className="flex flex-row items-center gap-1">
                        <User className="size-4" />
                        <div>{collection.owner_username || '--'}</div>
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
