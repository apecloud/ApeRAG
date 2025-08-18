import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { CollectionHeader } from '../collection-header';

export default async function Page({
  params,
}: Readonly<{
  params: Promise<{ collectionId: string }>;
}>) {
  const { collectionId } = await params;
  const serverApi = await getServerApi();
  const res = await serverApi.defaultApi.collectionsCollectionIdGet({
    collectionId,
  });
  const collection = res.data;

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          {
            title: 'Collections',
            href: '/workspace/collections',
          },
          {
            title: 'Knowledge Graph',
          },
        ]}
      />
      <CollectionHeader collection={collection} />
      <PageContent>Knowledge Graph</PageContent>
    </PageContainer>
  );
}
