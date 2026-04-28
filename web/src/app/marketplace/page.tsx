import {
  PageContainer,
  PageContent,
} from '@/components/page-container';
import { Button } from '@/components/ui/button';
import { listMarketplaceCollections } from '@/features/marketplace/server-api';
import type { SharedCollection } from '@/features/marketplace/types';
import { ENTITY_PALETTE } from '@/lib/design-tokens';
import { BookOpen, Plus, Star } from 'lucide-react';
import { getTranslations } from 'next-intl/server';
import Link from 'next/link';
import { CollectionList } from './collection-list';

export const dynamic = 'force-dynamic';

export default async function Page() {
  const page_marketplace = await getTranslations('page_marketplace');
  const sidebar_workspace = await getTranslations('sidebar_workspace');
  let collections: SharedCollection[] = [];
  try {
    const res = await listMarketplaceCollections({ page: 1, pageSize: 100 });
    collections = res.items || [];
  } catch (err) {
    console.log(err);
  }

  const featured = [...collections].sort(
    (a, b) => (b.subscription_count || 0) - (a.subscription_count || 0),
  )[0];

  return (
    <PageContainer>
      <PageContent className="max-w-7xl px-5 py-8 md:px-8 md:py-10">
        <div className="mb-8 flex flex-col gap-5 lg:flex-row lg:items-end">
          <div className="min-w-0 flex-1">
            <div className="text-muted-foreground font-mono text-[11px] tracking-[0.12em] uppercase">
              {page_marketplace('workspace_label')}
            </div>
            <h1 className="font-serif mt-2 text-4xl leading-none font-normal tracking-normal md:text-[44px]">
              {page_marketplace('metadata.title')}
            </h1>
            <p className="text-muted-foreground mt-3 max-w-2xl text-sm leading-6">
              {page_marketplace('metadata.description')}
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" asChild>
              <Link href="/workspace/collections">
                <BookOpen className="size-4" />
                {sidebar_workspace('collections')}
              </Link>
            </Button>
          </div>
        </div>

        {featured && (
          <div className="bg-foreground text-background mb-6 grid overflow-hidden rounded-xl border border-foreground/10 shadow-sm lg:grid-cols-[1.35fr_0.65fr]">
            <div className="p-6 md:p-8">
              <div className="text-primary font-mono text-[11px] tracking-[0.12em] uppercase">
                {page_marketplace('featured_label')}
              </div>
              <h2 className="font-serif mt-3 max-w-2xl text-3xl leading-tight font-normal md:text-[40px]">
                {featured.title}
              </h2>
              {featured.description && (
                <p className="text-background/70 mt-3 max-w-2xl text-sm leading-6">
                  {featured.description}
                </p>
              )}
              <div className="mt-6 flex flex-wrap items-center gap-6">
                <div>
                  <div className="font-mono text-2xl tabular-nums">
                    {featured.subscription_count || 0}
                  </div>
                  <div className="text-background/50 mt-1 font-mono text-[11px] tracking-[0.08em] uppercase">
                    {page_marketplace('subscriptions')}
                  </div>
                </div>
                <div className="text-background/60 text-sm">
                  {page_marketplace('published_by', {
                    owner:
                      featured.owner_username ||
                      page_marketplace('owner_unknown'),
                  })}
                </div>
              </div>
              <div className="mt-7 flex flex-wrap gap-2">
                <Button asChild>
                  <Link
                    href={`/marketplace/collections/${featured.id}/documents`}
                  >
                    <Star className="size-4" />
                    {page_marketplace('preview')}
                  </Link>
                </Button>
                <Button
                  variant="outline"
                  className="border-background/20 bg-background/5 text-background hover:bg-background/10 hover:text-background"
                  asChild
                >
                  <Link href="/workspace/collections">
                    <Plus className="size-4" />
                    {page_marketplace('publish_hint_action')}
                  </Link>
                </Button>
              </div>
            </div>
            <div className="relative hidden min-h-72 border-l border-background/10 lg:block">
              <div className="bg-primary/25 absolute top-1/2 left-1/2 size-56 -translate-x-1/2 -translate-y-1/2 rounded-full blur-3xl" />
              <div className="absolute inset-8">
                <MarketplaceGraphMotif />
              </div>
            </div>
          </div>
        )}

        <CollectionList collections={collections} />
      </PageContent>
    </PageContainer>
  );
}

const MarketplaceGraphMotif = () => {
  const nodes = [
    [50, 50, 16, ENTITY_PALETTE.event],
    [22, 22, 8, ENTITY_PALETTE.person],
    [78, 28, 9, ENTITY_PALETTE.org],
    [84, 76, 10, ENTITY_PALETTE.concept],
    [27, 78, 8, ENTITY_PALETTE.product],
    [52, 88, 7, ENTITY_PALETTE.doc],
  ] as const;

  return (
    <div className="relative h-full w-full">
      <svg
        aria-hidden="true"
        className="absolute inset-0 h-full w-full text-background/20"
        viewBox="0 0 100 100"
      >
        {nodes.slice(1).map(([x, y], index) => (
          <line
            key={`${x}-${y}-${index}`}
            x1="50"
            y1="50"
            x2={x}
            y2={y}
            stroke="currentColor"
            strokeWidth="0.45"
          />
        ))}
      </svg>
      {nodes.map(([x, y, size, color], index) => (
        <div
          key={`${x}-${y}`}
          className="absolute rounded-full border border-background/20 shadow-sm"
          style={{
            left: `${x}%`,
            top: `${y}%`,
            width: size,
            height: size,
            backgroundColor: color,
            transform: 'translate(-50%, -50%)',
          }}
          aria-hidden="true"
        >
          {index === 0 && (
            <div className="border-primary/60 absolute inset-[-10px] rounded-full border border-dashed" />
          )}
        </div>
      ))}
    </div>
  );
};
