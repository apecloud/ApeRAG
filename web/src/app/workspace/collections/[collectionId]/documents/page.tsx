import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { listDocuments } from '@/features/document/server-api';
import type { Document } from '@/features/document/types';
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
  const { page: parsedPage, pageSize: parsedPageSize } = parsePageParams({
    page,
    pageSize,
  });

  const page_collections = await getTranslations('page_collections');
  const page_documents = await getTranslations('page_documents');

  let documents: Document[] = [];
  let pageCount = 0;
  try {
    const data = await listDocuments(collectionId, {
      page: parsedPage,
      pageSize: parsedPageSize,
      sortBy: 'created',
      sortOrder: 'desc',
      search,
    });
    documents = data.items ?? [];
    pageCount = data.total_pages ?? 0;
  } catch (err) {
    console.log(err);
  }

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
        <DocumentsTable data={toJson(documents)} pageCount={pageCount} />
      </PageContent>
    </PageContainer>
  );
}
