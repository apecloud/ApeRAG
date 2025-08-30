import { CollectionGraph } from '@/app/workspace/collections/[collectionId]/graph/collection-graph';
import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { getTranslations } from 'next-intl/server';
import { CollectionHeader } from '../collection-header';

export default async function Page({
  params,
}: Readonly<{
  params: Promise<{ collectionId: string }>;
}>) {
  const { collectionId } = await params;
  const serverApi = await getServerApi();
  const page_marketplace = await getTranslations('page_marketplace');
  const page_graph = await getTranslations('page_graph');
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
            title: page_marketplace('metadata.title'),
            href: '/workspace/market/collections',
          },
          {
            title: collectionRes?.data.title,
          },
          {
            title: page_graph('metadata.title'),
          },
        ]}
      />
      <div className="flex h-[calc(100vh-48px)] flex-col px-0">
        <CollectionHeader collection={collectionRes.data} className="w-full" />
        <PageContent className="flex w-full flex-1 flex-col">
          <CollectionGraph marketplace={true} />
        </PageContent>
      </div>
    </PageContainer>
  );
}
