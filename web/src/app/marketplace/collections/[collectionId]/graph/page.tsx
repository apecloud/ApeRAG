import { CollectionGraph } from '@/app/workspace/collections/[collectionId]/graph/collection-graph';
import { PageContainer, PageContent } from '@/components/page-container';
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
      <div className="flex h-[calc(100vh-48px)] flex-col px-0">
        <CollectionHeader collection={collectionRes.data} className="w-full" />
        <PageContent className="flex w-full flex-1 flex-col">
          <CollectionGraph marketplace={true} />
        </PageContent>
      </div>
    </PageContainer>
  );
}
