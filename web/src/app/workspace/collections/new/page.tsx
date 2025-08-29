import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { Metadata } from 'next';
import { getTranslations } from 'next-intl/server';
import { CollectionForm } from '../collection-form';

export async function generateMetadata(): Promise<Metadata> {
  const page_collection_new = await getTranslations('page_collection_new');
  return {
    title: page_collection_new('metadata.title'),
    description: page_collection_new('metadata.description'),
  };
}

export default async function Page() {
  const page_collection_new = await getTranslations('page_collection_new');
  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          { title: 'Collections', href: '/workspace/collections' },
          { title: 'New' },
        ]}
      />
      <PageContent>
        <PageTitle>{page_collection_new('metadata.title')}</PageTitle>
        <PageDescription>
          {page_collection_new('metadata.description')}
        </PageDescription>
        <CollectionForm action="add" />
      </PageContent>
    </PageContainer>
  );
}
