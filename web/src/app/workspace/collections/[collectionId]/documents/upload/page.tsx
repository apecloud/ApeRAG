import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';

import { CollectionHeader } from '../../collection-header';
import { DocumentUpload } from './document-upload';

export default async function Page({
  params,
}: Readonly<{
  params: Promise<{ collectionId: string }>;
}>) {
  const { collectionId } = await params;

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
            title: 'Upload',
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
