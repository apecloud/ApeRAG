'use client';

import { SharedCollection } from '@/api';
import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { useState } from 'react';

export const CollectionList = ({
  collections,
}: {
  collections: SharedCollection[];
}) => {
  const [searchValue, setSearchValue] = useState<string>('');
  const page_marketplace = useTranslations('page_marketplace');
  if (collections.length === 0) {
    return (
      <div className="text-muted-foreground my-40 text-center">
        {page_marketplace('no_collections_found')}
      </div>
    );
  }

  return (
    <>
      <div className="mb-4">
        <Input
          placeholder={page_marketplace('search')}
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
            return (
              <Link
                key={collection.id}
                href={`/marketplace/collections/${collection.id}/documents`}
              >
                <Card className="hover:bg-accent/30 cursor-pointer gap-2 rounded-md">
                  <CardHeader className="px-4">
                    <CardTitle className="h-5 truncate">
                      {collection.title}
                    </CardTitle>
                  </CardHeader>
                  <CardDescription className="mb-4 truncate px-4">
                    {collection.description ||
                      page_marketplace('no_description_available')}
                  </CardDescription>
                </Card>
              </Link>
            );
          })}
      </div>
    </>
  );
};
