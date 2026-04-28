'use client';

import { FormatDate } from '@/components/format-date';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import type { CollectionView } from '@/features/collection/types';
import { cn } from '@/lib/utils';
import _ from 'lodash';
import {
  ArrowUpRight,
  Calendar,
  Circle,
  Database,
  Lock,
  Plus,
  Share2,
} from 'lucide-react';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { useMemo, useState } from 'react';

type CollectionFilter = 'all' | 'mine' | 'published' | 'subscribed';

const statusClasses: Record<string, string> = {
  ACTIVE: 'text-primary',
  INACTIVE: 'text-muted-foreground',
  DELETED: 'text-destructive',
};

export const CollectionList = ({
  collections,
}: {
  collections: CollectionView[];
}) => {
  const [filter, setFilter] = useState<CollectionFilter>('all');
  const page_collections = useTranslations('page_collections');
  const page_collection_new = useTranslations('page_collection_new');

  const stats = useMemo(() => {
    return {
      total: collections.length,
      owned: collections.filter((collection) => !collection.subscription_id)
        .length,
      published: collections.filter((collection) => collection.is_published)
        .length,
      subscribed: collections.filter((collection) => collection.subscription_id)
        .length,
    };
  }, [collections]);

  const filteredCollections = useMemo(() => {
    return collections.filter((collection) => {
      if (filter === 'mine' && collection.subscription_id) return false;
      if (filter === 'published' && !collection.is_published) return false;
      if (filter === 'subscribed' && !collection.subscription_id) return false;

      return true;
    });
  }, [collections, filter]);

  const filterOptions: Array<{
    key: CollectionFilter;
    label: string;
    count: number;
  }> = [
    {
      key: 'all',
      label: page_collections('filter_all'),
      count: stats.total,
    },
    {
      key: 'mine',
      label: page_collections('mine'),
      count: stats.owned,
    },
    {
      key: 'published',
      label: page_collections('filter_published'),
      count: stats.published,
    },
    {
      key: 'subscribed',
      label: page_collections('subscribed'),
      count: stats.subscribed,
    },
  ];

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center">
        <div className="bg-muted inline-flex w-fit flex-wrap gap-1 rounded-xl p-1">
          {filterOptions.map((option) => {
            const active = filter === option.key;
            return (
              <button
                key={option.key}
                type="button"
                onClick={() => setFilter(option.key)}
                className={cn(
                  'inline-flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm transition-colors',
                  active
                    ? 'bg-card text-foreground shadow-xs'
                    : 'text-muted-foreground hover:text-foreground',
                )}
              >
                <span>{option.label}</span>
                <span className="font-mono text-[11px] tabular-nums">
                  {option.count}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {collections.length === 0 ? (
        <EmptyCollections
          title={page_collections('no_collections_found')}
          description={page_collections('empty_description')}
          actionLabel={page_collection_new('metadata.title')}
        />
      ) : filteredCollections.length === 0 ? (
        <EmptyCollections
          title={page_collections('no_filter_results')}
          description={page_collections('no_filter_results_description')}
          actionLabel={page_collection_new('metadata.title')}
        />
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {filteredCollections.map((collection) => (
            <CollectionCard
              key={collection.id}
              collection={collection}
              statusLabel={_.upperFirst(
                _.lowerCase(collection.status ?? undefined),
              )}
            />
          ))}
        </div>
      )}
    </div>
  );
};

const EmptyCollections = ({
  title,
  description,
  actionLabel,
}: {
  title: string;
  description: string;
  actionLabel: string;
}) => {
  return (
    <div className="border-border/70 bg-card flex min-h-80 flex-col items-center justify-center rounded-xl border border-dashed px-6 py-16 text-center">
      <div className="bg-accent-soft text-accent-ink flex size-11 items-center justify-center rounded-full">
        <Database className="size-5" />
      </div>
      <div className="mt-4 text-base font-medium">{title}</div>
      <div className="text-muted-foreground mt-2 max-w-md text-sm leading-6">
        {description}
      </div>
      <Button asChild className="mt-6">
        <Link href="/workspace/collections/new">
          <Plus className="size-4" />
          {actionLabel}
        </Link>
      </Button>
    </div>
  );
};

const CollectionCard = ({
  collection,
  statusLabel,
}: {
  collection: CollectionView;
  statusLabel: string;
}) => {
  const page_collections = useTranslations('page_collections');
  const isSubscribed = Boolean(collection.subscription_id);
  const href = isSubscribed
    ? `/marketplace/collections/${collection.id}/documents`
    : `/workspace/collections/${collection.id}/documents`;
  const title = collection.title || page_collections('untitled_collection');
  const status = collection.status ?? 'INACTIVE';

  return (
    <Link href={href} target={isSubscribed ? '_blank' : '_self'}>
      <Card className="group hover:border-border hover:shadow-md min-h-56 cursor-pointer gap-0 overflow-hidden rounded-xl border-border/70 py-0 transition-all hover:-translate-y-0.5">
        <CardHeader className="gap-4 px-5 pt-5 pb-4">
          <div className="flex items-start gap-3">
            <div className="bg-accent-soft text-accent-ink flex size-10 shrink-0 items-center justify-center rounded-lg">
              <Database className="size-5" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex min-w-0 items-center gap-2">
                <CardTitle className="truncate text-[15px] leading-5 font-medium">
                  {title}
                </CardTitle>
                <ArrowUpRight className="text-muted-foreground group-hover:text-foreground size-3.5 shrink-0 transition-colors" />
              </div>
              <div className="text-muted-foreground mt-1 flex flex-wrap items-center gap-2 text-xs">
                <span className="flex items-center gap-1">
                  <Circle
                    className={cn(
                      'size-2 fill-current stroke-none',
                      statusClasses[status] ?? 'text-muted-foreground',
                    )}
                  />
                  {statusLabel}
                </span>
                {collection.owner_username && (
                  <span>{collection.owner_username}</span>
                )}
              </div>
            </div>
          </div>

          {collection.description && (
            <CardDescription className="line-clamp-2 text-[13px] leading-5">
              {collection.description}
            </CardDescription>
          )}
        </CardHeader>

        <CardFooter className="border-border/70 mt-auto justify-between border-t px-5 py-3 text-xs">
          <div className="text-muted-foreground flex min-w-0 items-center gap-2">
            <Calendar className="size-3.5 shrink-0" />
            <span className="truncate">
              {collection.updated || collection.created ? (
                <FormatDate
                  datetime={new Date(
                    collection.updated || collection.created || '',
                  )}
                />
              ) : (
                page_collections('no_date')
              )}
            </span>
          </div>
          {isSubscribed ? (
            <Badge variant="outline" className="bg-background gap-1 rounded-sm">
              <Share2 className="size-3" />
              {page_collections('subscribed')}
            </Badge>
          ) : collection.is_published ? (
            <Badge className="bg-accent-soft text-accent-ink border-accent-soft rounded-sm border">
              {page_collections('public')}
            </Badge>
          ) : (
            <Badge variant="secondary" className="gap-1 rounded-sm">
              <Lock className="size-3" />
              {page_collections('private')}
            </Badge>
          )}
        </CardFooter>
      </Card>
    </Link>
  );
};
