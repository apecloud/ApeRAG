import { PageContainer, PageContent } from '@/components/page-container';
import {
  getMarketplaceCollection,
  listMarketplaceCollectionDocuments,
} from '@/features/marketplace/server-api';
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
  const [collection, documents] = await Promise.all([
    getMarketplaceCollection(collectionId),
    listMarketplaceCollectionDocuments(collectionId, {
      ...parsePageParams({ page, pageSize }),
      sortBy: 'created',
      sortOrder: 'desc',
      search,
    }),
  ]);

  if (!collection) {
    notFound();
  }

  const documentItems = toJson(documents.items || []);
  const collectionJson = toJson(collection);

  return (
    <PageContainer>
      <CollectionHeader collection={collectionJson} />
      <PageContent className="max-w-7xl px-5 py-6 md:px-8">
        <DocumentsTable
          collection={collectionJson}
          data={documentItems}
          pageCount={documents.total_pages ?? 0}
        />
      </PageContent>
    </PageContainer>
  );
}
