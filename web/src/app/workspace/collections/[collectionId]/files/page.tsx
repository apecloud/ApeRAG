import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';
import { CollectionHeader } from '../collection-header';
import { FilesTable } from './files-table';

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
      <CollectionHeader collection={toJson(collectionRes.data)} />

      <PageContent>
        <FilesTable data={toJson(documentsRes.data.items || [])} />
      </PageContent>
    </PageContainer>
  );
}
