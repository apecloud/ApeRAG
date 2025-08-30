import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';

import { getServerApi } from '@/lib/api/server';

import { getTranslations } from 'next-intl/server';
import { notFound } from 'next/navigation';
import { EvaluationResult } from './evaluation-result';

export default async function Page({
  params,
}: {
  params: Promise<{ evaluationId: string }>;
}) {
  const { evaluationId } = await params;
  const serverApi = await getServerApi();
  const page_evaluation = await getTranslations('page_evaluation');
  let evaluation;

  try {
    const [evaluationRes] = await Promise.all([
      serverApi.evaluationApi.getEvaluationApiV1EvaluationsEvalIdGet({
        evalId: evaluationId,
      }),
    ]);
    evaluation = evaluationRes.data;
  } catch (err) {
    console.log(err);
  }

  if (!evaluation) {
    notFound();
  }

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          {
            title: page_evaluation('metadata.title'),
            href: '/workspace/evaluations',
          },
          { title: evaluation.name || '--' },
        ]}
      />
      <PageContent>
        <EvaluationResult evaluation={evaluation} />
      </PageContent>
    </PageContainer>
  );
}
