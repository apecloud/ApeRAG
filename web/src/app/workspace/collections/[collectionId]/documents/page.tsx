import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { parsePageParams, toJson } from '@/lib/utils';
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

  const [documentsRes] = await Promise.all([
    serverApi.defaultApi.collectionsCollectionIdDocumentsGet({
      collectionId,
      ...parsePageParams({ page, pageSize }),
      sortBy: 'created',
      sortOrder: 'desc',
      search,
    }),
  ]);

  //@ts-expect-error api define has a bug
  const documents = toJson(documentsRes.data.items || []);

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
          },
        ]}
      />
      <CollectionHeader />
      <PageContent>
        <DocumentsTable
          data={documents}
          pageCount={documentsRes.data.total_pages}
        />
      </PageContent>
    </PageContainer>
  );
}
