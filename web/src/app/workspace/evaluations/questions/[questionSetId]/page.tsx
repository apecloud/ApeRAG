import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { notFound } from 'next/navigation';
import { QuestionsList } from './questions-list';

export default async function Page({
  params,
}: {
  params: Promise<{ questionSetId: string }>;
}) {
  const { questionSetId } = await params;

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
          { title: 'Question Sets', href: `/workspace/evaluations/questions` },
          { title: questionSet.name ?? '--' },
        ]}
      />
      <PageContent>
        <PageTitle>Question Sets</PageTitle>
        <PageDescription className="mb-8">
          An Evaluation Question Set is a focused collection of questions
          designed to systematically assess the effectiveness, impact.
        </PageDescription>

        <QuestionsList questionSet={questionSet} />
      </PageContent>
    </PageContainer>
  );
}
