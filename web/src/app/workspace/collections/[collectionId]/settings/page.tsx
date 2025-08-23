import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { notFound } from 'next/navigation';
import { CollectionForm } from '../../collection-form';
import { CollectionHeader } from '../collection-header';

export default async function Page({
  params,
}: Readonly<{
  params: Promise<{ collectionId: string }>;
}>) {
  const { collectionId } = await params;
  const serverApi = await getServerApi();
  const res = await serverApi.defaultApi.collectionsCollectionIdGet({
    collectionId,
  });
  const collection = res.data;

  if (!collection) {
    notFound();
  }

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          {
            title: 'Collections',
            href: '/workspace/collections',
          },
          {
            title: 'Settings',
          },
        ]}
      />
      <CollectionHeader collection={collection} />
      <PageContent>
        <CollectionForm action="edit" collection={collection} />
      </PageContent>
    </PageContainer>
  );
}
