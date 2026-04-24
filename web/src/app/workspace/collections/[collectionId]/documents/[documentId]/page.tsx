import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import {
  getDocument,
  getDocumentPreview,
} from '@/features/document/server-api';
import { toJson } from '@/lib/utils';
import _ from 'lodash';
import { notFound } from 'next/navigation';
import { CollectionHeader } from '../../collection-header';
import { DocumentDetail } from './document-detail';

export default async function Page({
  params,
}: {
  params: Promise<{ collectionId: string; documentId: string }>;
}) {
  const { collectionId, documentId } = await params;

  const [document, documentPreview] = await Promise.all([
    getDocument(collectionId, documentId),
    getDocumentPreview(collectionId, documentId),
  ]);

  if (!document || !documentPreview) {
    notFound();
  }

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
            title: _.truncate(document?.name || '', { length: 30 }),
          },
        ]}
      />
      <CollectionHeader />
      <PageContent className="h-[100%] pt-4">
        <DocumentDetail
          document={toJson(document)}
          documentPreview={toJson(documentPreview)}
        />
      </PageContent>
    </PageContainer>
  );
}
