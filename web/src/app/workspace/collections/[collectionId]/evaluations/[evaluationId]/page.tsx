import { EvaluationRunDetail } from '@/components/evaluation/evaluation-run-detail';
import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import {
  getEvaluationRunDetail,
  listEvaluationRunItems,
} from '@/features/evaluation/server-api';
import { getTranslations } from 'next-intl/server';
import { CollectionHeader } from '../../collection-header';

export default async function Page({
  params,
}: Readonly<{
  params: Promise<{ collectionId: string; evaluationId: string }>;
}>) {
  const { collectionId, evaluationId } = await params;
  const [runDetail, runItems] = await Promise.all([
    getEvaluationRunDetail(evaluationId),
    listEvaluationRunItems(evaluationId),
  ]);

  const pageCollections = await getTranslations('page_collections');
  const pageEvaluations = await getTranslations('page_collection_evaluations');

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
            title:
              runDetail.payload?.run?.name ||
              runDetail.payload?.run?.id ||
              evaluationId,
          },
        ]}
      />
      <CollectionHeader />
      <PageContent>
        <EvaluationRunDetail
          botId={runDetail.payload?.run?.bot_id ?? undefined}
          detail={runDetail.payload}
          items={runItems.items}
          unavailable={runDetail.unavailable || runItems.unavailable}
          error={runDetail.error || runItems.error}
        />
      </PageContent>
    </PageContainer>
  );
}
