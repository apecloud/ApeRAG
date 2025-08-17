import { PageContainer } from '@/components/page-container';
import { PageHeader } from '@/components/page-header';
import { useIntl } from 'umi';

const QuestionSetListPage = () => {
  const { formatMessage } = useIntl();

  return (
    <PageContainer>
      <PageHeader title={formatMessage({ id: 'evaluation.question_sets' })} />
    </PageContainer>
  );
};

export default QuestionSetListPage;
