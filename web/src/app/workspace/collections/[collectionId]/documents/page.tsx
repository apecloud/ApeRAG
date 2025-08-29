import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { parsePageParams, toJson } from '@/lib/utils';
import { getTranslations } from 'next-intl/server';
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

  const page_collections = await getTranslations('page_collections');
  const page_documents = await getTranslations('page_documents');

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
            title: page_collections('metadata.title'),
            href: '/workspace/collections',
          },
          {
            title: page_documents('metadata.title'),
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
