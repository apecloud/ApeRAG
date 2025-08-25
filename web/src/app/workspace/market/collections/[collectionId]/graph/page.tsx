import { CollectionGraph } from '@/app/workspace/collections/[collectionId]/graph/collection-graph';
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
  const [collectionRes] = await Promise.all([
    serverApi.defaultApi.marketplaceCollectionsCollectionIdGet({
      collectionId,
    }),
  ]);

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          {
            title: 'Collections',
            href: '/workspace/market/collections',
          },
          {
            title: 'Knowledge Graph',
          },
        ]}
      />
      <CollectionHeader collection={collectionRes.data} />
      <PageContent>
        <CollectionGraph marketplace={true} />
      </PageContent>
    </PageContainer>
  );
}
