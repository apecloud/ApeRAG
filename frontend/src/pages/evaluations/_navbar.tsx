import { QuestionSet } from '@/api/models';
import { NAVIGATION_WIDTH } from '@/constants';
import { PlusOutlined } from '@ant-design/icons';
import { Button, Divider, Flex, Skeleton, theme, Typography } from 'antd';
import { Link, styled, useIntl, useLocation, useModel, useParams } from 'umi';

const { Title } = Typography;

const StyledNavbar = styled('div')`
  width: ${NAVIGATION_WIDTH}px;
  height: 100%;
  border-right: 1px solid ${(props) => props.theme.colorBorderSecondary};
  padding: 16px;
  display: flex;
  flex-direction: column;
`;

const StyledLink = styled(Link)`
  display: block;
  padding: 8px 12px;
  border-radius: 6px;
  color: ${(props) => props.theme.colorText};
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;

  &:hover {
    background-color: ${(props) => props.theme.controlItemBgHover};
  }

  &.active {
    background-color: ${(props) => props.theme.controlItemBgActive};
  }
`;

import { ArrowLeftOutlined } from '@ant-design/icons';

export const Navbar = () => {
  const { token } = theme.useToken();
  const { formatMessage } = useIntl();
  const location = useLocation();
  const params = useParams<{ questionSetId?: string; evaluationId?: string }>();
  const { questionSets, loading, getQuestionSet } = useModel('questionSet');
  const { currentEvaluation } = useModel('evaluation');
  const { questionSetId } = params;

  const currentQuestionSet = questionSetId
    ? getQuestionSet(questionSetId)
    : null;
  const isEvaluationDetailPage =
    location.pathname.match(/^\/evaluations\/(eval_.+)/);

  const renderQuestionSets = (
    sets: QuestionSet[] | undefined,
    isLoading: boolean,
  ) => {
    if (isLoading) {
      return <Skeleton active paragraph={{ rows: 4 }} />;
    }

    if (!sets || sets.length === 0) {
      return (
        <Flex
          style={{ color: token.colorTextTertiary, textAlign: 'center' }}
          flex={1}
          justify="center"
          align="center"
        >
          {formatMessage({ id: 'text.empty' })}
        </Flex>
      );
    }

    return sets.map((set) => (
      <StyledLink
        key={set.id}
        to={`/evaluations/question-sets/${set.id}`}
        theme={token}
        className={
          location.pathname.startsWith(
            `/evaluations/question-sets/${set.id}`,
          )
            ? 'active'
            : ''
        }
        title={set.name}
      >
        {set.name}
      </StyledLink>
    ));
  };

  const renderNavbarHeader = () => {
    let title = formatMessage({ id: 'evaluation.name' });
    let backLink = '/evaluations'; // Correct back link
    let showBackArrow = false;
    let entityName: string | undefined = undefined;

    if (isEvaluationDetailPage && currentEvaluation) {
      entityName = currentEvaluation.name;
      showBackArrow = true;
    } else if (currentQuestionSet) {
      entityName = currentQuestionSet.name;
      showBackArrow = true;
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

  return (
    <StyledNavbar theme={token}>
      <div style={{ marginBottom: 16, height: 32, display: 'flex', alignItems: 'center' }}>
        {renderNavbarHeader()}
      </div>
      <Divider style={{ margin: '0 0 16px 0' }} />
      <Flex vertical flex={1} style={{ overflow: 'hidden' }}>
        <Flex
          justify="space-between"
          align="center"
          style={{ marginBottom: 8 }}
        >
          <Title level={5} style={{ margin: 0 }}>
            {formatMessage({ id: 'evaluation.question_sets' })}
          </Title>
          <Link to="/evaluations/question-sets/new">
            <Button type="text" shape="circle" icon={<PlusOutlined />} />
          </Link>
        </Flex>
        <Flex
          vertical
          style={{
            overflowY: 'auto',
            flex: 1,
            paddingRight: 8,
            marginRight: -8,
          }}
        >
          {renderQuestionSets(questionSets, loading)}
        </Flex>
      </Flex>
    </StyledNavbar>
  );
};
