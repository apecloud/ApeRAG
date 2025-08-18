import { PageContainer } from '@/components/page-container';
import { PageHeader } from '@/components/page-header';
import { EvaluationApi } from '@/api/apis/evaluation-api';
import {
  App,
  Button,
  Flex,
  Popconfirm,
  Skeleton,
  Space,
  Table,
  Typography,
} from 'antd';
import { useIntl, useParams, history, useModel } from 'umi';
import { useRequest } from 'ahooks';
import { EditOutlined, DeleteOutlined } from '@ant-design/icons';

const { Paragraph } = Typography;

const QuestionSetReadOnlyPage = () => {
  const { formatMessage } = useIntl();
  const { message } = App.useApp();
  const { refresh: refreshQuestionSets } = useModel('questionSet');
  const params = useParams<{ questionSetId: string }>();
  const qsId = params.questionSetId;
  const evaluationApi = new EvaluationApi();

  const { data: questionSet, loading } = useRequest(
    () => {
      if (!qsId) return Promise.resolve(null);
      return evaluationApi.getQuestionSetApiV1QuestionSetsQsIdGet({ qsId });
    },
    {
      refreshDeps: [qsId],
      onSuccess: (res) => {
        if (!res?.data) {
          message.error(formatMessage({ id: 'tips.request.error' }));
        }
      },
    },
  );

  const handleDelete = async () => {
    if (!qsId) return;
    try {
      await evaluationApi.deleteQuestionSetApiV1QuestionSetsQsIdDelete({ qsId });
      message.success(formatMessage({ id: 'tips.delete.success' }));
      refreshQuestionSets();
      history.push('/evaluations/list');
    } catch (error) {
      message.error(formatMessage({ id: 'tips.delete.failed' }));
    }
  };

  const pageHeaderExtra = (
    <Space>
      <Button
        icon={<EditOutlined />}
        onClick={() => history.push(`/evaluations/question-sets/${qsId}/edit`)}
      >
        {formatMessage({ id: 'common.edit' })}
      </Button>
      <Popconfirm
        title={formatMessage({ id: 'evaluation.question_sets.delete.confirm' }, { name: questionSet?.data.name || '' })}
        onConfirm={handleDelete}
        okText={formatMessage({ id: 'evaluation.ok' })}
        cancelText={formatMessage({ id: 'evaluation.cancel' })}
      >
        <Button danger icon={<DeleteOutlined />}>
          {formatMessage({ id: 'common.delete' })}
        </Button>
      </Popconfirm>
    </Space>
  );

  return (
    <PageContainer>
      <Flex justify="space-between" align="center">
        <PageHeader
          title={
            loading ? (
              <Skeleton.Input active size="small" />
            ) : (
              questionSet?.data.name
            )
          }
          description={
            loading ? (
              <Skeleton active paragraph={{ rows: 1 }} />
            ) : (
              <Paragraph>{questionSet?.data.description}</Paragraph>
            )
          }
        />
        <div>{pageHeaderExtra}</div>
      </Flex>

      <Table
        loading={loading}
        columns={[
          {
            title: formatMessage({
              id: 'evaluation.question_sets.question',
            }),
            dataIndex: 'question_text',
            key: 'question',
            width: '50%',
          },
          {
            title: formatMessage({
              id: 'evaluation.question_sets.ground_truth',
            }),
            dataIndex: 'ground_truth',
            key: 'ground_truth',
            width: '50%',
          },
        ]}
        dataSource={questionSet?.data.questions || []}
        rowKey="id"
        locale={{
          emptyText: formatMessage({ id: 'text.empty' }),
        }}
        pagination={false}
      />
    </PageContainer>
  );
};

export default QuestionSetReadOnlyPage;
