import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { listSearches } from '@/features/retrieval/server-api';
import { getTranslations } from 'next-intl/server';
import { CollectionHeader } from '../collection-header';
import { SearchTable } from './search-table';

export default async function Page({
  params,
}: Readonly<{
  params: Promise<{ collectionId: string }>;
}>) {
  const page_collections = await getTranslations('page_collections');
  const page_search = await getTranslations('page_search');
  const { collectionId } = await params;

  const searchRes = await listSearches(collectionId);

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          {
            title: page_collections('metadata.title'),
            href: '/workspace/collections',
          },
          {
            title: page_search('metadata.title'),
          },
        ]}
      />
      <CollectionHeader />
      <PageContent>
        <SearchTable data={searchRes.items ?? []} />
      </PageContent>
    </PageContainer>
  );
}
