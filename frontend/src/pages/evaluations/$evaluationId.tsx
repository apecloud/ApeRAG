import { PageContainer, PageHeader } from '@/components';
import { UI_EVALUATION_STATUS } from '@/constants';
import { EvaluationItem, EvaluationItemStatus, EvaluationStatus } from '@/api/models';
import { SyncOutlined } from '@ant-design/icons';
import {
  App,
  Badge,
  Button,
  Card,
  Col,
  Divider,
  Dropdown,
  Popconfirm,
  Row,
  Skeleton,
  Space,
  Tag,
  theme,
  Tooltip,
  Typography,
} from 'antd';
import { useState, useEffect, useMemo } from 'react';
import { FormattedMessage, useIntl, useModel, useParams, Link, history } from 'umi';

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
  const { message } = App.useApp();
  const {
    currentEvaluation,
    loading,
    getEvaluation,
    deleteEvaluation,
    pauseEvaluation,
    resumeEvaluation,
    retryEvaluation,
  } = useModel('evaluation');
  const [isRefreshing, setIsRefreshing] = useState(false);

  const hasFailedItems = useMemo(() => {
    return currentEvaluation?.items?.some((item) => item.status === EvaluationItemStatus.FAILED) ?? false;
  }, [currentEvaluation]);

  const handleDelete = async () => {
    if (!evaluationId) return;
    const success = await deleteEvaluation(evaluationId);
    if (success) {
      message.success(formatMessage({ id: 'evaluation.delete.success' }));
      history.push('/evaluations');
    } else {
      message.error(formatMessage({ id: 'evaluation.delete.failure' }));
    }
  };

  const handlePause = async () => {
    if (!evaluationId) return;
    const success = await pauseEvaluation(evaluationId);
    if (success) {
      message.success(formatMessage({ id: 'evaluation.pause.success' }));
    } else {
      message.error(formatMessage({ id: 'evaluation.pause.failure' }));
    }
  };

  const handleResume = async () => {
    if (!evaluationId) return;
    const success = await resumeEvaluation(evaluationId);
    if (success) {
      message.success(formatMessage({ id: 'evaluation.resume.success' }));
    } else {
      message.error(formatMessage({ id: 'evaluation.resume.failure' }));
    }
  };

  const handleRetry = async (scope: 'failed' | 'all') => {
    if (!evaluationId) return;
    const success = await retryEvaluation(evaluationId, { scope });
    if (success) {
      message.success(formatMessage({ id: 'evaluation.retry.success' }));
    } else {
      message.error(formatMessage({ id: 'evaluation.retry.failure' }));
    }
  };

  useEffect(() => {
    if (evaluationId) {
      getEvaluation(evaluationId);
    }
  }, [evaluationId]);

  useEffect(() => {
    const isRefreshable =
      currentEvaluation?.status === EvaluationStatus.PENDING ||
      currentEvaluation?.status === EvaluationStatus.RUNNING;

    if (!isRefreshable) {
      return;
    }

    const intervalId = setInterval(() => {
      if (evaluationId) {
        const currentScrollY = window.scrollY;
        setIsRefreshing(true);
        getEvaluation(evaluationId, { background: true }).finally(() => {
          setIsRefreshing(false);
          window.scrollTo(0, currentScrollY);
        });
      }
    }, 30000); // 30 seconds

    return () => clearInterval(intervalId);
  }, [currentEvaluation?.status, evaluationId, getEvaluation]);

  if ((loading && !isRefreshing) || !currentEvaluation) {
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
    items,
    config,
    collection_name,
    question_set_name,
  } = currentEvaluation;

  const headerTitle = (
    <Title level={3} style={{ margin: 0 }}>
      {name}
    </Title>
  );

  const headerExtra = (
    <Space>
      <Button icon={<SyncOutlined spin={isRefreshing} />} onClick={() => getEvaluation(evaluationId!)} loading={loading && !isRefreshing}>
        <FormattedMessage id="action.refresh" />
      </Button>
      {status === EvaluationStatus.RUNNING && (
        <Button onClick={handlePause} loading={loading}>
          <FormattedMessage id="action.pause" />
        </Button>
      )}
      {status === EvaluationStatus.PAUSED && (
        <Button onClick={handleResume} loading={loading}>
          <FormattedMessage id="action.resume" />
        </Button>
      )}
      <Dropdown
        menu={{
          items: [
            {
              key: 'failed',
              label: formatMessage({ id: 'action.retryFailed' }),
              disabled: !hasFailedItems,
            },
            {
              key: 'all',
              label: formatMessage({ id: 'action.retryAll' }),
            },
          ],
          onClick: ({ key }) => handleRetry(key as 'failed' | 'all'),
        }}
      >
        <Button loading={loading}>
          <FormattedMessage id="action.retry" />
        </Button>
      </Dropdown>
      <Popconfirm
        title={formatMessage({ id: 'evaluation.delete.confirm.title' })}
        description={formatMessage({ id: 'evaluation.delete.confirm.description' })}
        onConfirm={handleDelete}
        okText={formatMessage({ id: 'action.yes' })}
        cancelText={formatMessage({ id: 'action.no' })}
      >
        <Button danger loading={loading}>
          <FormattedMessage id="action.delete" />
        </Button>
      </Popconfirm>
    </Space>
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
        <Space>
          <Badge
            status={status ? UI_EVALUATION_STATUS[status] : 'default'}
            text={
              <Text type="secondary">
                <FormattedMessage id={`evaluation.status.${status}`} />
              </Text>
            }
          />
          {headerExtra}
        </Space>
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

      <div>{items?.map(renderResultItem)}</div>
    </PageContainer>
  );
};
