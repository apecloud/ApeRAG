import { CollectionGraph } from '@/app/workspace/collections/[collectionId]/graph/collection-graph';
import { PageContainer, PageContent } from '@/components/page-container';
import { getMarketplaceCollection } from '@/features/marketplace/server-api';
import { notFound } from 'next/navigation';
import { CollectionHeader } from '../collection-header';

export default async function Page({
  params,
}: Readonly<{
  params: Promise<{ collectionId: string }>;
}>) {
  const { collectionId } = await params;
  const collection = await getMarketplaceCollection(collectionId);
  if (!collection) {
    notFound();
  }

  return (
    <PageContainer>
      <div className="flex h-[calc(100vh-48px)] flex-col px-0">
        <CollectionHeader collection={collection} className="w-full" />
        <PageContent className="flex w-full flex-1 flex-col">
          <CollectionGraph marketplace={true} />
        </PageContent>
      </div>
    </PageContainer>
  );
}
