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
import { notFound } from 'next/navigation';
import { CollectionHeader } from '../../collection-header';
import { DocumentDetail } from './document-detail';

const truncate = (value: string, length: number) => {
  if (value.length <= length) return value;
  return `${value.slice(0, Math.max(0, length - 3))}...`;
};

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
            title: truncate(document?.name || '', 30),
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
