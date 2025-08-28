import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { CollectionHeader } from '../collection-header';
import { SearchTable } from './search-table';

export default async function Page({
  params,
}: Readonly<{
  params: Promise<{ collectionId: string }>;
}>) {
  const { collectionId } = await params;
  const serverApi = await getServerApi();

  const [searchRes] = await Promise.all([
    serverApi.defaultApi.collectionsCollectionIdSearchesGet({
      collectionId,
    }),
  ]);

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          {
            title: 'Collections',
            href: '/workspace/collections',
          },
          {
            title: 'Search Effect',
          },
        ]}
      />
      <CollectionHeader />
      <PageContent>
        <SearchTable data={searchRes.data.items || []} />
      </PageContent>
    </PageContainer>
  );
}
