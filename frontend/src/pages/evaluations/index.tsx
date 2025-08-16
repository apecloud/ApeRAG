import { PageContainer, PageHeader, RefreshButton } from '@/components';
import { DATETIME_FORMAT, UI_EVALUATION_STATUS } from '@/constants';
import { ExperimentOutlined, PlusOutlined, SearchOutlined } from '@ant-design/icons';
import { useInterval } from 'ahooks';
import {
  Avatar,
  Badge,
  Button,
  Card,
  Col,
  Divider,
  Input,
  Result,
  Row,
  Space,
  theme,
  Tooltip,
  Typography,
} from 'antd';
import _ from 'lodash';
import moment from 'moment';
import { useEffect, useState } from 'react';
import { UndrawEmpty } from 'react-undraw-illustrations';
import { FormattedMessage, Link, useIntl, useModel } from 'umi';

export default () => {
  const [searchParams, setSearchParams] = useState<{
    name?: string;
  }>();
  const { token } = theme.useToken();
  const { formatMessage } = useIntl();
  const { evaluations, evaluationsLoading, getEvaluations } =
    useModel('evaluation');

  useInterval(() => {
    if (evaluations?.some((evaluation) => evaluation.status !== 'COMPLETED' && evaluation.status !== 'FAILED')) {
      getEvaluations();
    }
  }, 3000);

  const header = (
    <PageHeader
      title={formatMessage({ id: 'evaluation.name' })}
      description={formatMessage({ id: 'evaluation.tips' })}
    >
      <Space>
        <Input
          placeholder={formatMessage({ id: 'action.search' })}
          prefix={
            <Typography.Text disabled>
              <SearchOutlined />
            </Typography.Text>
          }
          onChange={(e) => {
            setSearchParams({ ...searchParams, name: e.currentTarget.value });
          }}
          allowClear
          value={searchParams?.name}
        />
        <Link to="/evaluations/new">
          <Tooltip title={<FormattedMessage id="evaluation.add" />}>
            <Button type="primary" icon={<PlusOutlined />} />
          </Tooltip>
        </Link>
        <RefreshButton
          loading={evaluationsLoading}
          onClick={() => getEvaluations()}
        />
      </Space>
    </PageHeader>
  );

  useEffect(() => {
    getEvaluations();
  }, []);

  if (evaluations === undefined) return;

  const _evaluations = evaluations?.filter((item) => {
    const nameMatch = searchParams?.name
      ? item.name?.includes(searchParams.name)
      : true;
    return nameMatch;
  });

  return (
    <PageContainer>
      {header}
      {_.isEmpty(_evaluations) ? (
        <Result
          icon={
            <UndrawEmpty primaryColor={token.colorPrimary} height="200px" />
          }
          subTitle={<FormattedMessage id="text.empty" />}
        />
      ) : (
        <Row gutter={[24, 24]}>
          {_evaluations?.map((evaluation) => {
            return (
              <Col
                key={evaluation.id}
                xs={24}
                sm={24}
                md={12}
                lg={8}
                xl={8}
                xxl={6}
              >
                <Link to={`/evaluations/${evaluation.id}`}>
                  <Card
                    size="small"
                    hoverable
                  >
                    <div
                      style={{ display: 'flex', gap: 12, alignItems: 'center' }}
                    >
                      <Avatar
                        size={40}
                        icon={<ExperimentOutlined />}
                        shape="square"
                        style={{
                          flex: 'none',
                          backgroundColor: '#1890ff',
                          color: '#fff'
                        }}
                      />
                      <div style={{ flex: 'auto', minWidth: 0 }}>
                        <div>
                          <Typography.Text ellipsis>
                            {evaluation.name}
                          </Typography.Text>
                        </div>
                      </div>
                    </div>
                    <Divider style={{ marginBlock: 12 }} />
                    <Row gutter={[16,16]}>
                      <Col span={12}>
                        <Typography.Text type="secondary" style={{ fontSize: '0.9em' }}>
                          <FormattedMessage id="evaluation.questionCount" />
                        </Typography.Text>
                        <div>
                          <Typography.Text>
                            {evaluation.completed_questions} / {evaluation.total_questions}
                          </Typography.Text>
                        </div>
                      </Col>
                      <Col span={12}>
                        <Typography.Text type="secondary" style={{ fontSize: '0.9em' }}>
                          <FormattedMessage id="evaluation.averageScore" />
                        </Typography.Text>
                        <div>
                          <Typography.Text>
                            {evaluation.average_score?.toFixed(2) ?? '-'}
                          </Typography.Text>
                        </div>
                      </Col>
                    </Row>
                    <Divider style={{ marginBlock: 12 }} />
                    <div
                      style={{
                        display: 'flex',
                        gap: 8,
                        justifyContent: 'space-between',
                        alignItems: 'center',
                      }}
                    >
                      <Typography.Text
                        ellipsis
                        type="secondary"
                        style={{ fontSize: '0.9em' }}
                      >
                        {moment(evaluation?.gmt_updated).format(DATETIME_FORMAT)}
                      </Typography.Text>
                      <Badge
                        status={
                          evaluation.status
                            ? UI_EVALUATION_STATUS[evaluation.status]
                            : 'default'
                        }
                        text={
                          <Typography.Text
                            type="secondary"
                            style={{ fontSize: '0.9em' }}
                          >
                            <FormattedMessage
                              id={`evaluation.status.${evaluation.status}`}
                            />
                          </Typography.Text>
                        }
                      />
                    </div>
                  </Card>
                </Link>
              </Col>
            );
          })}
        </Row>
      )}
    </PageContainer>
  );
};
