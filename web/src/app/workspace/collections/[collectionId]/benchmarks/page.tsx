import { BenchmarksPanel } from '@/components/evaluation/benchmarks-panel';
import { listBenchmarkDatasets } from '@/components/evaluation/server';
import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { getTranslations } from 'next-intl/server';
import { CollectionHeader } from '../collection-header';

export default async function Page({
  params,
}: Readonly<{
  params: Promise<{ collectionId: string }>;
}>) {
  const { collectionId } = await params;
  const pageCollections = await getTranslations('page_collections');
  const pageBenchmarks = await getTranslations('page_benchmarks');

  const datasets = await listBenchmarkDatasets(collectionId);

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          {
            title: pageCollections('metadata.title'),
            href: '/workspace/collections',
          },
          {
            title: pageBenchmarks('metadata.title'),
          },
        ]}
      />
      <CollectionHeader />
      <PageContent>
        <BenchmarksPanel
          collectionId={collectionId}
          items={datasets.items}
          unavailable={datasets.unavailable}
          error={datasets.error}
        />
      </PageContent>
    </PageContainer>
  );
}
