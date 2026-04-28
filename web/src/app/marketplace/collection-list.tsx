'use client';

import { useAppContext } from '@/components/providers/app-provider';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import {
  subscribeMarketplaceCollection,
  unsubscribeMarketplaceCollection,
} from '@/features/marketplace/client-api';
import type { SharedCollection } from '@/features/marketplace/types';
import { cn } from '@/lib/utils';
import {
  ArrowUpRight,
  Database,
  FileText,
  Network,
  Search,
  Share2,
  Sparkles,
  Star,
  User,
  VectorSquare,
  type LucideIcon,
} from 'lucide-react';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useMemo, useState, useTransition } from 'react';

type MarketplaceFilter = 'all' | 'subscribed' | 'mine';

export const CollectionList = ({
  collections,
}: {
  collections: SharedCollection[];
}) => {
  const { user, signIn } = useAppContext();
  const [searchValue, setSearchValue] = useState<string>('');
  const [filter, setFilter] = useState<MarketplaceFilter>('all');
  const page_marketplace = useTranslations('page_marketplace');

  const stats = useMemo(() => {
    return {
      total: collections.length,
      subscribed: collections.filter((collection) => collection.subscription_id)
        .length,
      mine: collections.filter(
        (collection) => collection.owner_user_id === user?.id,
      ).length,
    };
  }, [collections, user?.id]);

  const filteredCollections = useMemo(() => {
    const normalizedSearch = searchValue.trim().toLowerCase();
    return collections.filter((collection) => {
      if (filter === 'subscribed' && !collection.subscription_id) return false;
      if (filter === 'mine' && collection.owner_user_id !== user?.id) {
        return false;
      }

      if (!normalizedSearch) return true;
      const searchableText = [
        collection.title,
        collection.description,
        collection.owner_username,
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase();
      return searchableText.includes(normalizedSearch);
    });
  }, [collections, filter, searchValue, user?.id]);

  const filterOptions: Array<{
    key: MarketplaceFilter;
    label: string;
    count: number;
  }> = [
    {
      key: 'all',
      label: page_marketplace('filter_all'),
      count: stats.total,
    },
    {
      key: 'subscribed',
      label: page_marketplace('filter_subscribed'),
      count: stats.subscribed,
    },
    {
      key: 'mine',
      label: page_marketplace('filter_mine'),
      count: stats.mine,
    },
  ];

  return (
    <div className="flex flex-col gap-5">
      <div className="border-border/70 bg-card rounded-xl border p-4 shadow-sm">
        <div className="relative max-w-xl">
          <Search className="text-muted-foreground pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2" />
          <Input
            className="bg-background/70 h-10 rounded-lg pl-9"
            placeholder={page_marketplace('search')}
            value={searchValue}
            onChange={(e) => setSearchValue(e.currentTarget.value)}
          />
        </div>
      </div>

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

      {collections.length === 0 ? (
        <MarketplaceEmptyState
          title={page_marketplace('no_collections_found')}
          description={page_marketplace('empty_description')}
        />
      ) : filteredCollections.length === 0 ? (
        <MarketplaceEmptyState
          title={page_marketplace('no_filter_results')}
          description={page_marketplace('no_filter_results_description')}
        />
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {filteredCollections.map((collection) => (
            <MarketplaceCollectionCard
              key={collection.id}
              collection={collection}
              isOwner={collection.owner_user_id === user?.id}
              isSignedIn={Boolean(user)}
              onSignIn={signIn}
            />
          ))}
        </div>
      )}
    </div>
  );
};

const MarketplaceEmptyState = ({
  title,
  description,
}: {
  title: string;
  description: string;
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
    </div>
  );
};

const MarketplaceCollectionCard = ({
  collection,
  isOwner,
  isSignedIn,
  onSignIn,
}: {
  collection: SharedCollection;
  isOwner: boolean;
  isSignedIn: boolean;
  onSignIn: () => void;
}) => {
  const router = useRouter();
  const page_marketplace = useTranslations('page_marketplace');
  const page_collections = useTranslations('page_collections');
  const [isPending, startTransition] = useTransition();
  const isSubscribed = Boolean(collection.subscription_id);

  const handleSubscription = () => {
    if (!isSignedIn) {
      onSignIn();
      return;
    }
    startTransition(async () => {
      if (isSubscribed) {
        await unsubscribeMarketplaceCollection(collection.id);
      } else {
        await subscribeMarketplaceCollection(collection.id);
      }
      router.refresh();
    });
  };

  return (
    <Card className="group hover:border-border border-border/70 relative min-h-64 gap-0 overflow-hidden rounded-xl py-0 transition-all hover:-translate-y-0.5 hover:shadow-md">
      <CardHeader className="gap-4 px-5 pt-5 pb-3">
        <div className="absolute top-5 right-5">
          {isOwner ? (
            <Badge className="bg-accent-soft text-accent-ink border-accent-soft rounded-sm border">
              {page_collections('mine')}
            </Badge>
          ) : isSubscribed ? (
            <Badge variant="secondary" className="gap-1 rounded-sm">
              <Share2 className="size-3" />
              {page_collections('subscribed')}
            </Badge>
          ) : null}
        </div>
        <div className="flex items-start gap-3">
          <div className="bg-accent-soft text-accent-ink flex size-10 shrink-0 items-center justify-center rounded-lg">
            <Database className="size-5" />
          </div>
          <div className="min-w-0 flex-1 pr-16">
            <div className="flex min-w-0 items-center gap-2">
              <CardTitle className="truncate text-[15px] leading-5 font-medium">
                {collection.title}
              </CardTitle>
              <ArrowUpRight className="text-muted-foreground group-hover:text-foreground size-3.5 shrink-0 transition-colors" />
            </div>
            <div className="text-muted-foreground mt-1 flex items-center gap-1 text-xs">
              <User className="size-3.5" />
              <span className="truncate">
                {collection.owner_username || page_marketplace('owner_unknown')}
              </span>
            </div>
          </div>
        </div>

        {collection.description && (
          <CardDescription className="line-clamp-3 min-h-15 text-[13px] leading-5">
            {collection.description}
          </CardDescription>
        )}

        <CapabilityChips collection={collection} />

        <div className="text-muted-foreground flex items-center gap-2 text-xs">
          <Star className="size-3.5" />
          <span className="font-mono tabular-nums">
            {collection.subscription_count || 0}
          </span>
          <span>{page_marketplace('subscriptions')}</span>
        </div>
      </CardHeader>

      <CardFooter className="mt-auto flex-col items-stretch gap-3 px-5 pt-1 pb-4 text-xs">
        <div className="flex gap-2">
          <Button asChild variant="outline" className="flex-1">
            <Link href={`/marketplace/collections/${collection.id}/documents`}>
              {page_marketplace('preview')}
            </Link>
          </Button>
          {isOwner ? (
            <Button asChild variant="secondary" className="flex-1">
              <Link href={`/workspace/collections/${collection.id}/documents`}>
                {page_marketplace('open_workspace_collection')}
              </Link>
            </Button>
          ) : (
            <Button
              type="button"
              className="flex-1"
              disabled={isPending}
              onClick={handleSubscription}
            >
              <Star className="size-4" />
              {isSubscribed
                ? page_collections('subscribed')
                : page_collections('subscribe')}
            </Button>
          )}
        </div>
      </CardFooter>
    </Card>
  );
};

const CapabilityChips = ({ collection }: { collection: SharedCollection }) => {
  const page_marketplace = useTranslations('page_marketplace');
  type CapabilityChip = {
    key: string;
    label: string;
    Icon: LucideIcon;
  };
  const chips = [
    collection.config?.enable_vector && {
      key: 'vector',
      label: page_marketplace('capability_vector'),
      Icon: VectorSquare,
    },
    collection.config?.enable_fulltext && {
      key: 'fulltext',
      label: page_marketplace('capability_fulltext'),
      Icon: FileText,
    },
    collection.config?.enable_knowledge_graph && {
      key: 'graph',
      label: page_marketplace('capability_graph'),
      Icon: Network,
    },
  ].filter((chip): chip is CapabilityChip => Boolean(chip));

  return (
    <div className="flex flex-wrap gap-1.5">
      {chips.map((chip) => {
        const Icon = chip.Icon;
        return (
          <Badge
            key={chip.key}
            variant="outline"
            className="bg-background gap-1 rounded-sm"
          >
            <Icon className="size-3" />
            {chip.label}
          </Badge>
        );
      })}
      {chips.length === 0 && (
        <Badge variant="secondary" className="gap-1 rounded-sm">
          <Sparkles className="size-3" />
          {page_marketplace('capability_basic')}
        </Badge>
      )}
    </div>
  );
};
