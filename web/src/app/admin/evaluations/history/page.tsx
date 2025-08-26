import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';

export default async function Page() {
  const serverApi = await getServerApi();

  const [resEvaluations] = await Promise.all([
    serverApi.evaluationApi.listEvaluationsApiV1EvaluationsGet({
      page: 1,
      pageSize: 20,
    }),
  ]);

  console.log(resEvaluations);

  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: 'Evaluation history' }]} />
      <PageContent>
        <PageTitle>Evaluation history</PageTitle>
        <PageDescription className="mb-8">
          Efficiently track, manage, and review the historical performance of
          your Retrieval-Augmented Generation (RAG) evaluations all in one
          place.
        </PageDescription>

        <div className="flex flex-col gap-6"></div>
      </PageContent>
    </PageContainer>
  );
}
