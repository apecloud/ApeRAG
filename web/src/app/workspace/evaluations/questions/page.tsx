import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { getTranslations } from 'next-intl/server';
import { QuestionSetList } from './question-set-list';

export default async function Page() {
  const serverApi = await getServerApi();
  const page_question_set = await getTranslations('page_question_set');
  const [questionSetsRes] = await Promise.all([
    serverApi.evaluationApi.listQuestionSetsApiV1QuestionSetsGet({
      page: 1,
      pageSize: 100,
    }),
  ]);

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[{ title: page_question_set('metadata.title') }]}
      />
      <PageContent>
        <PageTitle>{page_question_set('metadata.title')}</PageTitle>
        <PageDescription className="mb-8">
          {page_question_set('metadata.description')}
        </PageDescription>

        <QuestionSetList questionSets={questionSetsRes.data.items || []} />
      </PageContent>
    </PageContainer>
  );
}
