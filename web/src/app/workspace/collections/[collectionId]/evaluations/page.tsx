import { CollectionRunsPanel } from '@/components/evaluation/collection-runs-panel';
import { EvaluationDatasetsPanel } from '@/components/evaluation/evaluation-datasets-panel';
import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import {
  listEvaluationDatasets,
  listEvaluationRuns,
} from '@/features/evaluation/server-api';
import { getTranslations } from 'next-intl/server';
import { CollectionHeader } from '../collection-header';

export default async function Page({
  params,
}: Readonly<{
  params: Promise<{ collectionId: string }>;
}>) {
  const { collectionId } = await params;
  const pageCollections = await getTranslations('page_collections');
  const pageEvaluations = await getTranslations('page_collection_evaluations');

  const [datasets, runs] = await Promise.all([
    listEvaluationDatasets(collectionId),
    listEvaluationRuns({ collectionId }),
  ]);

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          {
            title: pageCollections('metadata.title'),
            href: '/workspace/collections',
          },
          {
            title: pageEvaluations('metadata.title'),
          },
        ]}
      />
      <CollectionHeader />
      <PageContent className="flex flex-col gap-6">
        <EvaluationDatasetsPanel
          collectionId={collectionId}
          items={datasets.items}
          unavailable={datasets.unavailable}
          error={datasets.error}
        />
        <CollectionRunsPanel
          collectionId={collectionId}
          datasets={datasets.items}
          runs={runs.items}
          unavailable={runs.unavailable}
          error={runs.error}
        />
      </PageContent>
    </PageContainer>
  );
}
