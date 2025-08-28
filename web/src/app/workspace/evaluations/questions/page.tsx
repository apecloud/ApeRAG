import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { QuestionSetList } from './question-set-list';

export default async function Page() {
  const serverApi = await getServerApi();

  const [questionSetsRes] = await Promise.all([
    serverApi.evaluationApi.listQuestionSetsApiV1QuestionSetsGet({
      page: 1,
      pageSize: 100,
    }),
  ]);

  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: 'Question Sets' }]} />
      <PageContent>
        <PageTitle>Question Sets</PageTitle>
        <PageDescription className="mb-8">
          An Evaluation Question Set is a focused collection of questions
          designed to systematically assess the effectiveness, impact.
        </PageDescription>

        <QuestionSetList questionSets={questionSetsRes.data.items || []} />
      </PageContent>
    </PageContainer>
  );
}
