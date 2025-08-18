import { PageContainer, PageHeader, RefreshButton } from '@/components';
import { DATETIME_FORMAT } from '@/constants';
import { PlusOutlined, ReadOutlined } from '@ant-design/icons';
import {
  Avatar,
  Button,
  Card,
  Col,
  Divider,
  Result,
  Row,
  Space,
  theme,
  Tooltip,
  Typography,
  Pagination,
  Flex,
} from 'antd';
import _ from 'lodash';
import moment from 'moment';
import { useState } from 'react';
import { UndrawEmpty } from 'react-undraw-illustrations';
import { FormattedMessage, Link, useIntl } from 'umi';
import { EvaluationApi } from '@/api/apis/evaluation-api';
import { QuestionSet } from '@/api/models';
import { useRequest } from 'ahooks';

export default () => {
  const { token } = theme.useToken();
  const { formatMessage } = useIntl();
  const evaluationApi = new EvaluationApi();
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);

  const {
    data: questionSetData,
    loading,
    refresh,
  } = useRequest(
    () =>
      evaluationApi.listQuestionSetsApiV1QuestionSetsGet({
        page,
        pageSize,
      }),
    {
      refreshDeps: [page, pageSize],
    },
  );

  const questionSets = questionSetData?.data.items as QuestionSet[] | undefined;
  const total = questionSetData?.data.total ?? 0;

  const header = (
    <PageHeader
      title={formatMessage({ id: 'evaluation.question_sets' })}
      description={formatMessage({ id: 'evaluation.question_sets.tips' })}
    >
      <Space>
        <Link to="/evaluations/question-sets/new">
          <Tooltip title={<FormattedMessage id="evaluation.question_sets.add" />}>
            <Button type="primary" icon={<PlusOutlined />} />
          </Tooltip>
        </Link>
        <RefreshButton loading={loading} onClick={refresh} />
      </Space>
    </PageHeader>
  );

  return (
    <PageContainer>
      {header}
      {_.isEmpty(questionSets) && !loading ? (
        <Result
          icon={
            <UndrawEmpty primaryColor={token.colorPrimary} height="200px" />
          }
          subTitle={<FormattedMessage id="text.empty" />}
        />
      ) : (
        <>
          <Row gutter={[24, 24]}>
            {questionSets?.map((questionSet) => {
              return (
                <Col
                  key={questionSet.id}
                  xs={24}
                  sm={24}
                  md={12}
                  lg={8}
                  xl={8}
                  xxl={6}
                >
                  <Link to={`/evaluations/question-sets/${questionSet.id}`}>
                    <Card size="small" hoverable>
                      <div
                        style={{
                          display: 'flex',
                          gap: 12,
                          alignItems: 'center',
                        }}
                      >
                        <Avatar
                          size={40}
                          icon={<ReadOutlined />}
                          shape="square"
                          style={{
                            flex: 'none',
                            backgroundColor: '#1890ff',
                            color: '#fff',
                          }}
                        />
                        <div style={{ flex: 'auto', minWidth: 0 }}>
                          <div>
                            <Typography.Text ellipsis>
                              {questionSet.name}
                            </Typography.Text>
                          </div>
                          <Typography.Text
                            type="secondary"
                            ellipsis
                            style={{ fontSize: '0.9em' }}
                          >
                            {questionSet.description || '-'}
                          </Typography.Text>
                        </div>
                      </div>
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
                          {moment(questionSet?.gmt_updated).format(
                            DATETIME_FORMAT,
                          )}
                        </Typography.Text>
                      </div>
                    </Card>
                  </Link>
                </Col>
              );
            })}
          </Row>
          <Flex justify="flex-end" style={{ marginTop: 24 }}>
            <Pagination
              current={page}
              pageSize={pageSize}
              total={total}
              onChange={(p, ps) => {
                setPage(p);
                setPageSize(ps);
              }}
              showSizeChanger
            />
          </Flex>
        </>
      )}
    </PageContainer>
  );
};
