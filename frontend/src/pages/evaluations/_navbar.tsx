import { NAVIGATION_WIDTH } from '@/constants';
import { Button, Divider, Flex, Menu, theme, Typography } from 'antd';
import { Link, styled, useIntl, useLocation, useModel, useParams } from 'umi';
import { ExperimentOutlined, ReadOutlined } from '@ant-design/icons';

const { Title } = Typography;

const StyledNavbar = styled('div')`
  width: ${NAVIGATION_WIDTH}px;
  height: 100%;
  border-right: 1px solid ${(props) => props.theme.colorBorderSecondary};
  padding: 16px;
  display: flex;
  flex-direction: column;
`;


import { ArrowLeftOutlined } from '@ant-design/icons';

export const Navbar = () => {
  const { token } = theme.useToken();
  const { formatMessage } = useIntl();
  const location = useLocation();
  const params = useParams<{ questionSetId?: string; evaluationId?: string }>();
  const { getQuestionSet } = useModel('questionSet');
  const { currentEvaluation } = useModel('evaluation');
  const { questionSetId } = params;

  const currentQuestionSet = questionSetId
    ? getQuestionSet(questionSetId)
    : null;
  const isEvaluationDetailPage =
    location.pathname.match(/^\/evaluations\/(eval_.+)/);

  const renderNavbarHeader = () => {
    let title = formatMessage({ id: 'evaluation.name' });
    let backLink = '/evaluations';
    let showBackArrow = false;
    let entityName: string | undefined = undefined;

    if (isEvaluationDetailPage && currentEvaluation) {
      entityName = currentEvaluation.name;
      showBackArrow = true;
    } else if (currentQuestionSet) {
      entityName = currentQuestionSet.name;
      showBackArrow = true;
      backLink = '/evaluations/question-sets';
    }

    if (showBackArrow) {
      return (
        <Flex align="center" gap={8}>
          <Link to={backLink}>
            <Button type="text" shape="circle" icon={<ArrowLeftOutlined />} />
          </Link>
          <Title
            level={5}
            style={{
              margin: 0,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
            title={entityName}
          >
            {entityName || '...'}
          </Title>
        </Flex>
      );
    }

    return (
      <Title level={5} style={{ margin: 0 }}>
        {title}
      </Title>
    );
  };

  const getSelectedKeys = () => {
    if (location.pathname.startsWith('/evaluations/question-sets')) {
      return ['question-sets'];
    }
    if (location.pathname.startsWith('/evaluations')) {
      return ['evaluations'];
    }
    return [];
  };

  return (
    <StyledNavbar theme={token}>
      <div
        style={{
          marginBottom: 16,
          height: 32,
          display: 'flex',
          alignItems: 'center',
        }}
      >
        {renderNavbarHeader()}
      </div>
      <Divider style={{ margin: '0 0 16px 0' }} />
      <Menu
        style={{
          border: 'none',
          background: 'transparent',
          width: 'calc(100% + 32px)',
          marginLeft: -16,
          marginRight: -16,
        }}
        selectedKeys={getSelectedKeys()}
        items={[
          {
            key: 'evaluations',
            icon: <ExperimentOutlined />,
            label: (
              <Link to="/evaluations">
                {formatMessage({ id: 'evaluation.name' })}
              </Link>
            ),
          },
          {
            key: 'question-sets',
            icon: <ReadOutlined />,
            label: (
              <Link to="/evaluations/question-sets">
                {formatMessage({ id: 'evaluation.question_sets' })}
              </Link>
            ),
          },
        ]}
      />
    </StyledNavbar>
  );
};
