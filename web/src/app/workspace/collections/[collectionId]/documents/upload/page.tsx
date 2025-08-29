import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';

import { getTranslations } from 'next-intl/server';
import { CollectionHeader } from '../../collection-header';
import { DocumentUpload } from './document-upload';

export default async function Page({
  params,
}: Readonly<{
  params: Promise<{ collectionId: string }>;
}>) {
  const { collectionId } = await params;
  const page_collections = await getTranslations('page_collections');
  const page_documents = await getTranslations('page_documents');

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          {
            title: page_collections('metadata.title'),
            href: '/workspace/collections',
          },
          {
            title: page_documents('metadata.title'),
            href: `/workspace/collections/${collectionId}/documents`,
          },
          {
            title: page_documents('upload'),
          },
        ]}
      />
      <CollectionHeader />
      <PageContent>
        <DocumentUpload />
      </PageContent>
    </PageContainer>
  );
}
