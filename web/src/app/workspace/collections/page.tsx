import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { Button } from '@/components/ui/button';

import { listCollections } from '@/features/collection/server-api';
import type { CollectionView } from '@/features/collection/types';
import { toJson } from '@/lib/utils';
import { Plus } from 'lucide-react';
import { Metadata } from 'next';
import { getTranslations } from 'next-intl/server';
import Link from 'next/link';
import { CollectionList } from './collection-list';

export const dynamic = 'force-dynamic';

export async function generateMetadata(): Promise<Metadata> {
  const page_collections = await getTranslations('page_collections');
  return {
    title: page_collections('metadata.title'),
    description: page_collections('metadata.description'),
  };
}

export default async function Page() {
  const page_collections = await getTranslations('page_collections');
  const page_collection_new = await getTranslations('page_collection_new');

  let collections: CollectionView[] = [];
  try {
    const data = await listCollections({
      page: 1,
      pageSize: 100,
      includeSubscribed: true,
    });
    collections = data.items ?? [];
  } catch (err) {
    console.log(err);
  }

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[{ title: page_collections('metadata.title') }]}
      />
      <PageContent className="max-w-7xl px-5 py-8 md:px-8 md:py-10">
        <div className="mb-9 flex flex-col gap-5 md:flex-row md:items-start md:justify-between">
          <div className="min-w-0">
            <div className="text-muted-foreground font-mono text-[11px] tracking-[0.12em] uppercase">
              {page_collections('workspace_label')}
            </div>
            <h1 className="font-serif mt-2 text-4xl leading-none font-normal tracking-normal md:text-[44px]">
              {page_collections('metadata.title')}
            </h1>
            <p className="text-muted-foreground mt-3 max-w-2xl text-sm leading-6">
              {page_collections('metadata.description')}
            </p>
          </div>
          <Button asChild className="w-fit shrink-0">
            <Link href="/workspace/collections/new">
              <Plus className="size-4" />
              {page_collection_new('metadata.title')}
            </Link>
          </Button>
        </div>
        <CollectionList collections={toJson(collections)} />
      </PageContent>
    </PageContainer>
  );
}
