import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { getTranslations } from 'next-intl/server';
import { EvaluationList } from './evaluation-list';

export default async function Page() {
  const serverApi = await getServerApi();
  const page_evaluation = await getTranslations('page_evaluation');
  const [resEvaluations] = await Promise.all([
    serverApi.evaluationApi.listEvaluationsApiV1EvaluationsGet({
      page: 1,
      pageSize: 100,
    }),
  ]);

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[{ title: page_evaluation('metadata.title') }]}
      />
      <PageContent>
        <PageTitle>{page_evaluation('metadata.title')}</PageTitle>
        <PageDescription className="mb-8">
          {page_evaluation('metadata.description')}
        </PageDescription>

        <EvaluationList evaluations={resEvaluations.data.items || []} />
      </PageContent>
    </PageContainer>
  );
}
