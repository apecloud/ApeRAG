import { PageContainer, PageContent } from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';
import { CollectionHeader } from '../../collection-header';
import { DocumentDetail } from './document-detail';

export default async function Page({
  params,
}: {
  params: Promise<{ collectionId: string; documentId: string }>;
}) {
  const { collectionId, documentId } = await params;
  const serverApi = await getServerApi();

  const [colelctionRes, documentPreviewRes] = await Promise.all([
    serverApi.defaultApi.marketplaceCollectionsCollectionIdGet({
      collectionId,
    }),
    serverApi.defaultApi.marketplaceCollectionsCollectionIdDocumentsDocumentIdPreviewGet(
      {
        collectionId,
        documentId,
      },
    ),
  ]);

  const documentPreview = toJson(documentPreviewRes.data);
  const collection = toJson(colelctionRes.data);

  return (
    <PageContainer>
      <CollectionHeader collection={collection} />
      <PageContent className="h-[100%]">
        <DocumentDetail documentPreview={documentPreview} />
      </PageContent>
    </PageContainer>
  );
}
