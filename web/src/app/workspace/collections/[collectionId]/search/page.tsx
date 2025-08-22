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

  const [collectionRes, searchRes] = await Promise.all([
    serverApi.defaultApi.collectionsCollectionIdGet({
      collectionId,
    }),
    serverApi.defaultApi.collectionsCollectionIdSearchesGet({
      collectionId,
    }),
  ]);

  const collection = collectionRes.data;

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          {
            title: 'Collections',
            href: '/workspace/collections',
          },
          {
            title: 'Experience Search',
          },
        ]}
      />
      <CollectionHeader collection={collection} />
      <PageContent>
        <SearchTable
          collection={collection}
          data={searchRes.data.items || []}
        />
      </PageContent>
    </PageContainer>
  );
}
