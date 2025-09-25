import { CollectionView } from '@/api';
import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';

import { Button } from '@/components/ui/button';
import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';
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
  const serverApi = await getServerApi();
  const page_collections = await getTranslations('page_collections');
  const page_marketplace = await getTranslations('page_marketplace');

  let collections: CollectionView[] = [];
  try {
    const res = await serverApi.defaultApi.collectionsGet({
      page: 1,
      pageSize: 100,
      includeSubscribed: true,
    });
    collections = res.data.items || [];
  } catch (err) {
    console.log(err);
  }

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[{ title: page_collections('metadata.title') }]}
      />
      <PageContent>
        <div className="flex flex-row items-center justify-between">
          <PageTitle>{page_collections('metadata.title')}</PageTitle>
          <Button variant="secondary" asChild>
            <Link href="/marketplace/collections">
              {page_marketplace('metadata.title')}
            </Link>
          </Button>
        </div>
        <PageDescription>
          {page_collections('metadata.description')}
        </PageDescription>
        <CollectionList collections={toJson(collections)} />
      </PageContent>
    </PageContainer>
  );
}
