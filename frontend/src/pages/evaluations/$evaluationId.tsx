import { PageContainer, PageHeader } from '@/components';
import { UI_EVALUATION_STATUS } from '@/constants';
import { EvaluationItem, EvaluationItemStatus } from '@/api/models';
import { SyncOutlined } from '@ant-design/icons';
import {
  Badge,
  Button,
  Card,
  Col,
  Divider,
  Row,
  Skeleton,
  Space,
  Tag,
  theme,
  Tooltip,
  Typography,
} from 'antd';
import { useState, useEffect } from 'react';
import { FormattedMessage, useIntl, useModel, useParams, Link } from 'umi';

const { Text, Paragraph, Title } = Typography;

const ExpandableText = ({ text, maxChars = 200 }: { text: string; maxChars?: number }) => {
  const { formatMessage } = useIntl();
  const [isExpanded, setIsExpanded] = useState(false);

  if (!text) {
    return <Paragraph type="secondary">N/A</Paragraph>;
  }

  if (text.length <= maxChars) {
    return <Paragraph style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{text}</Paragraph>;
  }

  return (
    <div>
      <Paragraph style={{ whiteSpace: 'pre-wrap', margin: 0 }}>
        {isExpanded ? text : `${text.substring(0, maxChars)}...`}
      </Paragraph>
      <Button type="link" onClick={() => setIsExpanded(!isExpanded)} style={{ padding: 0 }}>
        {isExpanded ? formatMessage({ id: 'action.showLess' }) : formatMessage({ id: 'action.showMore' })}
      </Button>
    </div>
  );
};

const ResultItemStatus = ({ item }: { item: EvaluationItem }) => {
  const { token } = theme.useToken();
  const { formatMessage } = useIntl();

  if (item.status === EvaluationItemStatus.RUNNING) {
    return <Tag icon={<SyncOutlined spin />} color="processing">{formatMessage({ id: 'evaluation.item.status.RUNNING' })}</Tag>;
  }

  if (item.status === EvaluationItemStatus.FAILED) {
    return <Tag color="error">{formatMessage({ id: 'evaluation.item.status.FAILED' })}</Tag>;
  }

  if (item.status === EvaluationItemStatus.COMPLETED) {
    if (item.llm_judge_score === null || item.llm_judge_score === undefined) {
      return <Tag color="warning">{formatMessage({ id: 'evaluation.item.noScore' })}</Tag>;
    }
    const scoreColor =
      item.llm_judge_score >= 4
        ? token.colorSuccess
        : item.llm_judge_score >= 3
        ? token.colorWarning
        : token.colorError;
    return (
      <Tag color={scoreColor} style={{ fontSize: 16, padding: '4px 10px', minWidth: 40, textAlign: 'center' }}>
        {item.llm_judge_score}
      </Tag>
    );
  }

  return <Tag>{formatMessage({ id: 'evaluation.item.status.PENDING' })}</Tag>;
};


export default () => {
  const { evaluationId } = useParams<{ evaluationId: string }>();
  const { formatMessage } = useIntl();
  const { currentEvaluation, loading, getEvaluation } = useModel('evaluation');

  useEffect(() => {
    if (evaluationId) {
      getEvaluation(evaluationId);
    }
  }, [evaluationId, getEvaluation]);

  if (loading || !currentEvaluation) {
    return (
      <PageContainer>
        <PageHeader title={<Skeleton.Input active size="small" />} />
        <Skeleton active style={{ marginTop: 24 }} />
      </PageContainer>
    );
  }

  const {
    name,
    status,
    average_score,
    results,
    config,
    collection_name,
    question_set_name,
  } = currentEvaluation;

  const headerTitle = (
    <Title level={3} style={{ margin: 0 }}>
      {name}
    </Title>
  );

  const renderResultItem = (item: EvaluationItem) => {
    return (
      <Card key={item.id} style={{ marginBottom: 16 }}>
        <Row gutter={16} align="top">
          <Col flex="auto">
            <Text strong>{item.question_text}</Text>
          </Col>
          <Col flex="none">
            <ResultItemStatus item={item} />
          </Col>
        </Row>
        <Divider style={{ margin: '12px 0' }} />
        <Space direction="vertical" style={{ width: '100%' }} size="middle">
          <div>
            <Text type="secondary">{formatMessage({ id: 'evaluation.detail.groundTruth' })}</Text>
            <ExpandableText text={item.ground_truth!} />
          </div>
          <div>
            <Text type="secondary">{formatMessage({ id: 'evaluation.detail.ragAnswer' })}</Text>
            <ExpandableText text={item.rag_answer!} />
          </div>
          <div>
            <Text type="secondary">{formatMessage({ id: 'evaluation.detail.judgeReasoning' })}</Text>
            <ExpandableText text={item.llm_judge_reasoning!} />
          </div>
        </Space>
      </Card>
    );
  };

  return (
    <PageContainer>
      <PageHeader title={headerTitle}>
        <Badge
          status={status ? UI_EVALUATION_STATUS[status] : 'default'}
          text={
            <Text type="secondary">
              <FormattedMessage id={`evaluation.status.${status}`} />
            </Text>
          }
        />
      </PageHeader>
      <Card style={{ marginTop: 24, marginBottom: 24 }}>
        <Row gutter={[32, 16]}>
          <Col>
            <Text type="secondary">{formatMessage({ id: 'evaluation.averageScore' })}</Text>
            <div>
              <Text style={{ fontSize: 24, lineHeight: '28px' }}>
                {average_score?.toFixed(2) ?? '-'}
              </Text>
            </div>
          </Col>
          <Col>
            <Text type="secondary">{formatMessage({ id: 'evaluation.detail.collection' })}</Text>
            <div>
              <Tooltip title={`ID: ${config?.collection_id}`}>
                <Link to={`/collections/${config?.collection_id}/documents`}>
                  {collection_name}
                </Link>
              </Tooltip>
            </div>
          </Col>
          <Col>
            <Text type="secondary">{formatMessage({ id: 'evaluation.detail.questionSet' })}</Text>
            <div>
              <Tooltip title={`ID: ${config?.question_set_id}`}>
                <Link to={`/evaluations/question-sets/${config?.question_set_id}`}>
                  {question_set_name}
                </Link>
              </Tooltip>
            </div>
          </Col>
        </Row>
      </Card>

      <div>{results?.map(renderResultItem)}</div>
    </PageContainer>
  );
};
