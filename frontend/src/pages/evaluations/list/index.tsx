import { PageContainer } from '@/components/page-container';
import { PageHeader } from '@/components/page-header';
import { useIntl } from 'umi';

const EvaluationListPage = () => {
  const { formatMessage } = useIntl();

  return (
    <PageContainer width="auto">
      <PageHeader title={formatMessage({ id: 'evaluation.list' })} />
    </PageContainer>
  );
};

export default EvaluationListPage;
