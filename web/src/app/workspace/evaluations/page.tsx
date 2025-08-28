import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { EvaluationList } from './evaluation-list';

export default async function Page() {
  const serverApi = await getServerApi();

  const [resEvaluations] = await Promise.all([
    serverApi.evaluationApi.listEvaluationsApiV1EvaluationsGet({
      page: 1,
      pageSize: 100,
    }),
  ]);

  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: 'Evaluation' }]} />
      <PageContent>
        <PageTitle>Evaluation</PageTitle>
        <PageDescription className="mb-8">
          Efficiently track, manage, and review the historical performance of
          your Retrieval-Augmented Generation (RAG) evaluations all in one
          place.
        </PageDescription>

        <EvaluationList evaluations={resEvaluations.data.items || []} />
      </PageContent>
    </PageContainer>
  );
}
