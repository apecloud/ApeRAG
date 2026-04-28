import { PageContainer, PageContent } from '@/components/page-container';
import { Button } from '@/components/ui/button';
import { listMarketplaceCollections } from '@/features/marketplace/server-api';
import type { SharedCollection } from '@/features/marketplace/types';
import { BookOpen } from 'lucide-react';
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

  return (
    <PageContainer>
      <PageContent className="max-w-7xl px-5 py-8 md:px-8 md:py-10">
        <div className="mb-8 flex flex-col gap-5 lg:flex-row lg:items-end">
          <div className="min-w-0 flex-1">
            <div className="text-muted-foreground font-mono text-[11px] tracking-[0.12em] uppercase">
              {page_marketplace('workspace_label')}
            </div>
            <h1 className="mt-2 font-serif text-4xl leading-none font-normal tracking-normal md:text-[44px]">
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

        <CollectionList collections={collections} />
      </PageContent>
    </PageContainer>
  );
}
