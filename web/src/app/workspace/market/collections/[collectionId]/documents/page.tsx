import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';
import { notFound } from 'next/navigation';
import { CollectionHeader } from '../collection-header';
import { DocumentsTable } from './documents-table';
// import { DocumentsTable } from './documents-table';

export default async function Page({
  params,
}: Readonly<{
  params: Promise<{ collectionId: string }>;
}>) {
  const { collectionId } = await params;
  const serverApi = await getServerApi();
  const [collectionRes, documentsRes] = await Promise.all([
    serverApi.defaultApi.marketplaceCollectionsCollectionIdGet({
      collectionId,
    }),
    serverApi.defaultApi.marketplaceCollectionsCollectionIdDocumentsGet({
      collectionId,
    }),
  ]);

  //@ts-expect-error api define has a bug
  const documents = toJson(documentsRes.data.items || []);
  const collection = collectionRes.data;

  if (!collection) {
    notFound();
  }

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          {
            title: 'Marketplace',
            href: '/workspace/market/collections',
          },
          {
            title: collection.title,
          },
        ]}
      />
      <CollectionHeader collection={collection} />
      <PageContent>
        <DocumentsTable
          collection={collection}
          data={documents}
          pageCount={documentsRes.data.total_pages}
        />
      </PageContent>
    </PageContainer>
  );
}
