import { PageContainer, PageContent } from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { parsePageParams, toJson } from '@/lib/utils';
import { notFound } from 'next/navigation';
import { CollectionHeader } from '../collection-header';
import { DocumentsTable } from './documents-table';

export default async function Page({
  params,
  searchParams,
}: Readonly<{
  params: Promise<{ collectionId: string }>;
  searchParams: Promise<{ page?: string; pageSize?: string; search?: string }>;
}>) {
  const { collectionId } = await params;
  const { page, pageSize, search } = await searchParams;
  const serverApi = await getServerApi();
  const [collectionRes, documentsRes] = await Promise.all([
    serverApi.defaultApi.marketplaceCollectionsCollectionIdGet({
      collectionId,
    }),
    serverApi.defaultApi.marketplaceCollectionsCollectionIdDocumentsGet({
      collectionId,
      ...parsePageParams({ page, pageSize }),
      sortBy: 'created',
      sortOrder: 'desc',
      search,
    }),
  ]);

  //@ts-expect-error api define has a bug
  const documents = toJson(documentsRes.data.items || []);
  const collection = toJson(collectionRes.data);

  if (!collection) {
    notFound();
  }

  return (
    <PageContainer>
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
