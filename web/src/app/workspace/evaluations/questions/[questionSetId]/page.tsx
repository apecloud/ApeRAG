import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { getTranslations } from 'next-intl/server';
import { notFound } from 'next/navigation';
import { QuestionsList } from './questions-list';

export default async function Page({
  params,
}: {
  params: Promise<{ questionSetId: string }>;
}) {
  const { questionSetId } = await params;
  const page_question_set = await getTranslations('page_question_set');

  const serverApi = await getServerApi();

  let questionSet;

  try {
    const [questionSetRes] = await Promise.all([
      serverApi.evaluationApi.getQuestionSetApiV1QuestionSetsQsIdGet({
        qsId: questionSetId,
      }),
    ]);
    questionSet = questionSetRes.data;
  } catch (err) {
    console.log(err);
  }

  if (!questionSet) {
    notFound();
  }

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          {
            title: page_question_set('metadata.title'),
            href: `/workspace/evaluations/questions`,
          },
          { title: questionSet.name ?? '--' },
        ]}
      />
      <PageContent>
        <PageTitle>{page_question_set('metadata.title')}</PageTitle>
        <PageDescription className="mb-8">
          {page_question_set('metadata.description')}
        </PageDescription>

        <QuestionsList questionSet={questionSet} />
      </PageContent>
    </PageContainer>
  );
}
