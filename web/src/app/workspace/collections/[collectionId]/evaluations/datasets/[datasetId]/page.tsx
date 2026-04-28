import { EvaluationDatasetItemsPanel } from '@/components/evaluation/evaluation-dataset-items-panel';
import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import {
  getEvaluationDataset,
  listEvaluationDatasetItems,
} from '@/features/evaluation/server-api';
import { getTranslations } from 'next-intl/server';
import { notFound } from 'next/navigation';
import { CollectionHeader } from '../../../collection-header';

export default async function Page({
  params,
}: Readonly<{
  params: Promise<{ collectionId: string; datasetId: string }>;
}>) {
  const { collectionId, datasetId } = await params;
  const [datasetState, itemsState] = await Promise.all([
    getEvaluationDataset(datasetId),
    listEvaluationDatasetItems(datasetId),
  ]);

  if (
    !datasetState.payload &&
    !datasetState.unavailable &&
    !datasetState.error
  ) {
    notFound();
  }

  const pageCollections = await getTranslations('page_collections');
  const pageEvaluations = await getTranslations('page_collection_evaluations');
  const dataset = datasetState.payload;

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
            href: `/workspace/collections/${collectionId}/evaluations`,
          },
          {
            title: dataset?.name || datasetId,
          },
        ]}
      />
      <CollectionHeader />
      <PageContent>
        {dataset ? (
          <EvaluationDatasetItemsPanel
            collectionId={collectionId}
            dataset={dataset}
            items={itemsState.items}
            unavailable={itemsState.unavailable}
            error={itemsState.error}
          />
        ) : null}
      </PageContent>
    </PageContainer>
  );
}
