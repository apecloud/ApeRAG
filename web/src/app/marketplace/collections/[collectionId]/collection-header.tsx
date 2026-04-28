'use client';

import { PageContent } from '@/components/page-container';
import { useAppContext } from '@/components/providers/app-provider';
import { Badge } from '@/components/ui/badge';
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb';
import { Button } from '@/components/ui/button';
import {
  subscribeMarketplaceCollection,
  unsubscribeMarketplaceCollection,
} from '@/features/marketplace/client-api';
import type { SharedCollection } from '@/features/marketplace/types';
import { cn } from '@/lib/utils';
import {
  BookOpen,
  Files,
  Network,
  Share2,
  Star,
  User,
} from 'lucide-react';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { useCallback, useMemo, useTransition } from 'react';

export const CollectionHeader = ({
  className,
  collection,
}: {
  className?: string;
  collection: SharedCollection;
}) => {
  const router = useRouter();
  const pathname = usePathname();
  const [isPending, startTransition] = useTransition();

  const { user, signIn } = useAppContext();
  const page_collections = useTranslations('page_collections');
  const page_documents = useTranslations('page_documents');
  const page_marketplace = useTranslations('page_marketplace');
  const page_graph = useTranslations('page_graph');
  const sidebar_workspace = useTranslations('sidebar_workspace');

  const isOwner = useMemo(
    () => collection.owner_user_id === user?.id,
    [collection.owner_user_id, user?.id],
  );
  const isSubscriber = useMemo(
    () => Boolean(collection.subscription_id),
    [collection.subscription_id],
  );

  const handleSubscribe = useCallback(() => {
    if (!user) {
      signIn();
      return;
    }

    startTransition(async () => {
      if (isSubscriber) {
        await unsubscribeMarketplaceCollection(collection.id);
      } else {
        await subscribeMarketplaceCollection(collection.id);
      }
      router.refresh();
    });
  }, [collection.id, isSubscriber, router, signIn, user]);

  return (
    <PageContent
      className={cn('max-w-7xl px-5 pt-6 pb-0 md:px-8', className)}
    >
      <div className="mb-4 flex flex-col gap-3 md:flex-row md:items-center">
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink asChild>
                <Link
                  href="/marketplace"
                  className="text-muted-foreground hover:text-foreground flex items-center gap-1"
                >
                  {page_marketplace('metadata.title')}
                </Link>
              </BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem className="max-w-64 truncate">
              {collection.title}
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>

        <Button variant="outline" className="md:ml-auto" asChild>
          <Link href="/workspace/collections">
            <BookOpen className="size-4" />
            {sidebar_workspace('collections')}
          </Link>
        </Button>
      </div>

      <div className="border-border/70 bg-card overflow-hidden rounded-xl border shadow-sm">
        <div className="grid gap-5 p-5 md:p-6 lg:grid-cols-[1fr_auto] lg:items-start">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <Badge className="bg-accent-soft text-accent-ink border-accent-soft rounded-sm border">
                {page_marketplace('published_collection')}
              </Badge>
              {isOwner && <Badge variant="secondary">{page_collections('mine')}</Badge>}
              {isSubscriber && !isOwner && (
                <Badge variant="secondary" className="gap-1">
                  <Share2 className="size-3" />
                  {page_collections('subscribed')}
                </Badge>
              )}
            </div>
            <h1 className="font-serif mt-3 text-3xl leading-tight font-normal md:text-[40px]">
              {collection.title}
            </h1>
            {collection.description && (
              <p className="text-muted-foreground mt-3 max-w-3xl text-sm leading-6">
                {collection.description}
              </p>
            )}
            <div className="text-muted-foreground mt-4 flex flex-wrap items-center gap-4 text-sm">
              <span className="flex items-center gap-1.5">
                <User className="size-4" />
                {collection.owner_username ||
                  page_marketplace('owner_unknown')}
              </span>
              <span className="flex items-center gap-1.5">
                <Star className="size-4" />
                <span className="font-mono tabular-nums">
                  {collection.subscription_count || 0}
                </span>
                {page_marketplace('subscriptions')}
              </span>
            </div>
          </div>

          <div className="flex flex-wrap gap-2 lg:justify-end">
            {isOwner ? (
              <Button asChild>
                <Link href={`/workspace/collections/${collection.id}/documents`}>
                  {page_marketplace('open_workspace_collection')}
                </Link>
              </Button>
            ) : (
              <Button
                type="button"
                disabled={isPending}
                onClick={handleSubscribe}
              >
                <Star className="size-4" />
                {isSubscriber
                  ? page_collections('subscribed')
                  : page_collections('subscribe')}
              </Button>
            )}
          </div>
        </div>

        <div className="border-border/70 bg-muted/60 flex flex-wrap gap-1 border-t px-3 py-2">
          <TabLink
            active={Boolean(
              pathname.match(
                `/marketplace/collections/${collection.id}/documents`,
              ),
            )}
            href={`/marketplace/collections/${collection.id}/documents`}
            icon={<Files className="size-4" />}
            label={page_documents('metadata.title')}
          />

          {collection.config?.enable_knowledge_graph && (
            <TabLink
              active={Boolean(
                pathname.match(
                  `/marketplace/collections/${collection.id}/graph`,
                ),
              )}
              href={`/marketplace/collections/${collection.id}/graph`}
              icon={<Network className="size-4" />}
              label={page_graph('metadata.title')}
            />
          )}
        </div>
      </div>
    </PageContent>
  );
};

const TabLink = ({
  active,
  href,
  icon,
  label,
}: {
  active: boolean;
  href: string;
  icon: React.ReactNode;
  label: string;
}) => {
  return (
    <Button
      asChild
      data-active={active}
      className="data-[active=true]:bg-card data-[active=true]:text-foreground data-[active=true]:shadow-xs text-muted-foreground rounded-lg"
      variant="ghost"
    >
      <Link href={href}>
        {icon}
        <span>{label}</span>
      </Link>
    </Button>
  );
};
