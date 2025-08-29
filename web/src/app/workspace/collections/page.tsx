import { CollectionView } from '@/api';
import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';

import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';
import { Metadata } from 'next';
import { getTranslations } from 'next-intl/server';
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
        <PageTitle>{page_collections('metadata.title')}</PageTitle>
        <PageDescription>
          {page_collections('metadata.description')}
        </PageDescription>
        <CollectionList collections={toJson(collections)} />
      </PageContent>
    </PageContainer>
  );
}
