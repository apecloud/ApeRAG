import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { getTranslations } from 'next-intl/server';
import { CollectionHeader } from '../collection-header';
import { EvaluationList } from './evaluation-list';

export default async function Page({
  params,
}: {
  params: Promise<{ collectionId: string }>;
}) {
  const { collectionId } = await params;
  const serverApi = await getServerApi();
  const page_evaluation = await getTranslations('page_evaluation');
  const page_collections = await getTranslations('page_collections');
  const [resEvaluations] = await Promise.all([
    serverApi.evaluationApi.listEvaluationsApiV1EvaluationsGet({
      collectionId,
      page: 1,
      pageSize: 100,
    }),
  ]);

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          {
            title: page_collections('metadata.title'),
            href: '/workspace/collections',
          },
          { title: page_evaluation('metadata.title') },
        ]}
      />
      <CollectionHeader />

      <PageContent>
        <EvaluationList evaluations={resEvaluations.data.items || []} />
      </PageContent>
    </PageContainer>
  );
}
