import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';
import _ from 'lodash';
import { CollectionHeader } from '../../collection-header';
import { DocumentDetail } from './document-detail';

export default async function Page({
  params,
}: {
  params: Promise<{ collectionId: string; documentId: string }>;
}) {
  const { collectionId, documentId } = await params;
  const serverApi = await getServerApi();

  const [documentRes, documentPreviewRes] = await Promise.all([
    serverApi.defaultApi.collectionsCollectionIdDocumentsDocumentIdGet({
      collectionId,
      documentId,
    }),
    serverApi.defaultApi.getDocumentPreview({
      collectionId,
      documentId,
    }),
  ]);

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
            title: 'Documents',
            href: `/workspace/collections/${collectionId}/documents`,
          },
          {
            title: _.truncate(document.name || '', { length: 30 }),
          },
        ]}
      />
      <CollectionHeader />
      <PageContent className="h-[100%]">
        <DocumentDetail document={document} documentPreview={documentPreview} />
      </PageContent>
    </PageContainer>
  );
}
