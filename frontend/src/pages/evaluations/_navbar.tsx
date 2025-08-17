import { QuestionSet } from '@/api/models';
import { NAVIGATION_WIDTH } from '@/constants';
import { PlusOutlined } from '@ant-design/icons';
import { Button, Divider, Flex, Skeleton, theme, Typography } from 'antd';
import { Link, styled, useIntl, useLocation, useModel } from 'umi';

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

export const Navbar = () => {
  const { token } = theme.useToken();
  const { formatMessage } = useIntl();
  const location = useLocation();
  const { questionSets, loading } = useModel('questionSet');

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

  return (
    <StyledNavbar theme={token}>
      <Flex vertical>
        <Title level={5}>{formatMessage({ id: 'evaluation.name' })}</Title>
        <StyledLink
          to="/evaluations/list"
          theme={token}
          className={location.pathname === '/evaluations/list' ? 'active' : ''}
        >
          {formatMessage({ id: 'evaluation.list' })}
        </StyledLink>
      </Flex>
      <Divider />
      <Flex vertical flex={1} style={{ overflow: 'hidden' }}>
        <Flex justify="space-between" align="center">
          <Title level={5}>
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
