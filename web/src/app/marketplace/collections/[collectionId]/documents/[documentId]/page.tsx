import { PageContainer, PageContent } from '@/components/page-container';
import {
  getMarketplaceCollection,
  getMarketplaceCollectionDocumentPreview,
} from '@/features/marketplace/server-api';
import { toJson } from '@/lib/utils';
import { notFound } from 'next/navigation';
import { CollectionHeader } from '../../collection-header';
import { DocumentDetail } from './document-detail';

export default async function Page({
  params,
}: {
  params: Promise<{ collectionId: string; documentId: string }>;
}) {
  const { collectionId, documentId } = await params;
  const [collection, documentPreview] = await Promise.all([
    getMarketplaceCollection(collectionId),
    getMarketplaceCollectionDocumentPreview(collectionId, documentId),
  ]);
  if (!collection || !documentPreview) {
    notFound();
  }

  return (
    <PageContainer>
      <CollectionHeader collection={collection} />
      <PageContent className="h-[100%]">
        <DocumentDetail documentPreview={toJson(documentPreview)} />
      </PageContent>
    </PageContainer>
  );
}
