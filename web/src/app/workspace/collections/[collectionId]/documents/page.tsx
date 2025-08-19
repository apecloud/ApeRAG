import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';
import { CollectionHeader } from '../collection-header';
import { DocumentsTable } from './documents-table';

export default async function Page({
  params,
}: Readonly<{
  params: Promise<{ collectionId: string }>;
}>) {
  const { collectionId } = await params;
  const serverApi = await getServerApi();

  const [collectionRes, documentsRes] = await Promise.all([
    serverApi.defaultApi.collectionsCollectionIdGet({
      collectionId,
    }),
    serverApi.defaultApi.collectionsCollectionIdDocumentsGet({
      collectionId,
    }),
  ]);

  const collection = toJson(collectionRes.data);

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
          },
        ]}
      />
      <CollectionHeader collection={collection} />

      <PageContent>
        <DocumentsTable
          collection={collection}
          data={toJson(documentsRes.data.items || [])}
        />
      </PageContent>
    </PageContainer>
  );
}
