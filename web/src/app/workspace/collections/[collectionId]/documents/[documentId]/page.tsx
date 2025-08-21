import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
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

  const [collectionRes, documentRes, documentPreviewRes] = await Promise.all([
    serverApi.defaultApi.collectionsCollectionIdGet({
      collectionId,
    }),
    serverApi.defaultApi.collectionsCollectionIdDocumentsDocumentIdGet({
      collectionId,
      documentId,
    }),
    serverApi.defaultApi.getDocumentPreview({
      collectionId,
      documentId,
    }),
  ]);

  const collection = toJson(collectionRes.data);
  const document = toJson(documentRes.data);
  const documentPreview = toJson(documentPreviewRes.data);

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          {
            title: 'Collections',
            href: '/workspace/collections',
          },
          {
            title: 'File Explorer',
            href: `/workspace/collections/${collection.id}/documents`,
          },
          {
            title: document.name || '',
          },
        ]}
      />
      <CollectionHeader collection={collection} />
      <PageContent className="h-[100%]">
        <DocumentDetail
          collection={collection}
          document={document}
          documentPreview={documentPreview}
        />
      </PageContent>
    </PageContainer>
  );
}
