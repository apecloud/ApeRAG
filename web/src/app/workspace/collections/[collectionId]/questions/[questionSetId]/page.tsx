import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { getTranslations } from 'next-intl/server';
import { notFound } from 'next/navigation';
import { CollectionHeader } from '../../collection-header';
import { QuestionsList } from './questions-list';

export default async function Page({
  params,
}: {
  params: Promise<{ questionSetId: string; collectionId: string }>;
}) {
  const { questionSetId, collectionId } = await params;
  const page_question_set = await getTranslations('page_question_set');
  const page_collections = await getTranslations('page_collections');

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
            title: page_collections('metadata.title'),
            href: '/workspace/collections',
          },
          {
            title: page_question_set('metadata.title'),
            href: `/workspace/collections/${collectionId}/questions`,
          },
          { title: questionSet.name || '--' },
        ]}
      />
      <CollectionHeader />
      <PageContent>
        <QuestionsList questionSet={questionSet} />
      </PageContent>
    </PageContainer>
  );
}
