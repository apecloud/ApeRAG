import {
  ChatMessage,
  Feedback,
  FeedbackTagEnum,
  FeedbackTypeEnum,
} from '@/api';
import { ApeMarkdown } from '@/components';
import { TypingAnimate } from '@/components/typing-animate';
import { DATETIME_FORMAT } from '@/constants';
import {
  BulbOutlined,
  CaretRightOutlined,
  DislikeFilled,
  LikeFilled,
  UserOutlined,
} from '@ant-design/icons';
import { useHover } from 'ahooks';
import {
  Alert,
  Avatar,
  Badge,
  Button,
  Collapse,
  CollapseProps,
  Drawer,
  GlobalToken,
  Input,
  Modal,
  Radio,
  Space,
  theme,
  Tooltip,
  Typography,
} from 'antd';
import _ from 'lodash';
import moment from 'moment';
import { useCallback, useMemo, useRef, useState } from 'react';
import { BsRobot } from 'react-icons/bs';
import { css, styled, useIntl } from 'umi';

export const StyledMessage = styled('div').withConfig({
  shouldForwardProp: (prop) => !['isAi'].includes(prop),
})<{ isAi: boolean }>`
  ${({ isAi }) => {
    return css`
      display: flex;
      flex-direction: row;
      justify-content: ${!isAi ? 'flex-end' : 'flex-start'};
      margin-bottom: 24px;
      gap: 12px;
      min-width: 240px;
    `;
  }}
`;

export const StyledMessageAvatar = styled(Avatar).withConfig({
  shouldForwardProp: (prop) => !['token'].includes(prop),
})<{ token: GlobalToken }>`
  ${({ token }) => {
    return css`
      background: ${token.colorPrimary};
      margin-top: 8px;
      box-shadow: ${token.boxShadow};
    `;
  }}
`;

export const StyledMessageBody = styled('div')`
  ${() => {
    return css`
      max-width: 75%;
      min-width: 25%;
      display: flex;
      flex-direction: column;
      gap: 4px;
    `;
  }}
`;

export const StyledMessageContent = styled('div').withConfig({
  shouldForwardProp: (prop) => !['isAi', 'token'].includes(prop),
})<{ isAi: boolean; token: GlobalToken }>`
  ${({ isAi, token }) => {
    return css`
      background: ${!isAi ? token.colorPrimary : token.colorBgContainer};
      color: ${!isAi ? token.colorWhite : token.colorText};
      min-height: 54px;
      padding: 16px;
      border-radius: 8px;
      box-shadow: ${token.boxShadow};
    `;
  }}
`;

export const StyledMessageInfo = styled('div').withConfig({
  shouldForwardProp: (prop) => !['token', 'isAi'].includes(prop),
})<{ token: GlobalToken; isAi: boolean }>`
  ${({ token, isAi }) => {
    return css`
      color: ${token.colorTextSecondary};
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      gap: 16px;
      align-items: center;
      min-height: 32px;
      justify-content: ${!isAi ? 'flex-end' : 'flex-start'};
    `;
  }}
`;

export const CollapseResult = ({
  title,
  children,
}: {
  title: string;
  children?: React.ReactNode;
}) => {
  const { token } = theme.useToken();
  const [open, setOpen] = useState<boolean>(false);

  return (
    <div style={{ marginBottom: 8 }}>
      <Space
        style={{
          background: token.controlItemBgHover,
          cursor: 'pointer',
          padding: '4px 8px',
          borderRadius: 100,
          alignItems: 'center',
          fontSize: '0.75rem',
        }}
        onClick={() => setOpen(!open)}
      >
        <CaretRightOutlined
          style={{
            transitionDuration: '0.3s',
            transform: `rotate(${open ? 90 : 0}deg)`,
          }}
        />
        {title}
      </Space>
      {open && (
        <div
          style={{
            padding: 12,
            border: `1px dashed ${token.colorBorder}`,
            borderRadius: 4,
            background: token.colorBgLayout,
            marginTop: 4,
          }}
        >
          {children}
        </div>
      )}
    </div>
  );
};

export const ChatMessageItem = ({
  parts,
  onVote = () => {},
  isAi,
}: {
  parts: ChatMessage[];
  isAi: boolean;
  onVote?: (item: ChatMessage, feedback: Feedback) => void;
}) => {
  const { token } = theme.useToken();
  const hoverRef = useRef(null);
  const isHovering = useHover(hoverRef);
  const { formatMessage } = useIntl();
  const [referencesVisible, setReferencesVisible] = useState<boolean>(false);
  const [feedbackModalVisible, setFeedbackModalVisible] =
    useState<boolean>(false);
  const [feedbackTag, setFeedbackTag] = useState<FeedbackTagEnum>();
  const [feedbackMessage, setFeedbackMessage] = useState<string>();

  const parseToolCallTitle = (content: string): { title: string; body: string } => {
    const lines = content.split('\n');
    const firstLine = lines[0] || '';
    
    // Check if first line has **title** format
    const titleMatch = firstLine.match(/^\*\*(.*?)\*\*$/);
    if (titleMatch) {
      const title = titleMatch[1].trim();
      const body = lines.slice(1).join('\n').trim();
      return { title, body };
    }
    
    // Fallback to default title
    return { title: 'Tool call', body: content };
  };

  const getReferences: () => CollapseProps['items'] = () =>
    parts
      .find((p) => p.type === 'references')
      ?.references?.map((reference, index) => ({
        key: index,
        label: (
          <Typography.Text
            style={{ maxWidth: 400, color: token.colorPrimary }}
            ellipsis
          >
            {index + 1}.{' '}
            {reference.metadata?.name ||
              reference.metadata?.source ||
              reference.metadata?.query ||
              reference.metadata?.type}
          </Typography.Text>
        ),
        children: (
          <div
            style={{
              borderTop: `1px solid ${token.colorBorderSecondary}`,
              paddingTop: 16,
            }}
          >
            <ApeMarkdown>{reference.text}</ApeMarkdown>
          </div>
        ),
        extra: (
          <Space>
            <BulbOutlined />
            {reference.score}
          </Space>
        ),
        style: {
          marginBottom: 24,
          borderRadius: token.borderRadiusLG,
          border: `1px solid ${token.colorBorderSecondary}`,
        },
      }));

  const partReference = useMemo(() => {
    return parts.find((p) => p.type === 'references');
  }, [parts]);

  const handleFeedback = useCallback(
    (type: FeedbackTypeEnum) => {
      if (!partReference) return;

      if (type === partReference?.feedback?.type) {
        onVote(partReference, {});
      } else if (type === FeedbackTypeEnum.good) {
        onVote(partReference, { type });
      } else {
        setFeedbackModalVisible(true);
      }
    },
    [partReference],
  );

  const handleFeedbackSubmit = useCallback(() => {
    if (!partReference) return;
    onVote(partReference, {
      type: FeedbackTypeEnum.bad,
      tag: feedbackTag,
      message: feedbackMessage,
    });

    setFeedbackModalVisible(false);
    setFeedbackTag(undefined);
    setFeedbackMessage(undefined);
  }, [partReference, feedbackTag, feedbackMessage]);

  return (
    <>
      <StyledMessage isAi={isAi}>
        {isAi && (
          <StyledMessageAvatar src={<BsRobot />} size="large" token={token} />
        )}
        <StyledMessageBody ref={hoverRef}>
          <StyledMessageContent token={token} isAi={isAi}>
            {!isAi
              ? parts[0].data
              : parts.map((part, index) => {
                  switch (part.type) {
                    case 'tool_call_result': {
                      const { title, body } = parseToolCallTitle(part.data || '');
                      return (
                        <CollapseResult key={index} title={title}>
                          <ApeMarkdown>{body}</ApeMarkdown>
                        </CollapseResult>
                      );
                    }
                    case 'thinking':
                      return (
                        <CollapseResult key={index} title="Thinking">
                          <ApeMarkdown>{part.data}</ApeMarkdown>
                        </CollapseResult>
                      );
                    case 'message':
                      return <ApeMarkdown key={index}>{part.data}</ApeMarkdown>;
                    case 'error':
                      return (
                        <Alert key={index} message={part.data} type="error" />
                      );
                    case 'start':
                      return (
                        parts.length === 1 && <TypingAnimate key={index} />
                      );
                    case 'stop':
                    case 'welcome':
                    case 'references':
                      return '';
                    default:
                      return 'unknow part type';
                  }
                })}
          </StyledMessageContent>
          <StyledMessageInfo token={token} isAi={isAi}>
            {moment(
              parts?.[0]?.timestamp ? parts?.[0]?.timestamp * 1000 : undefined,
            ).format(DATETIME_FORMAT)}
            {!_.isEmpty(partReference?.references) && isAi && (
              <Tooltip title={formatMessage({ id: 'text.references' })}>
                <Button
                  type="text"
                  onClick={() => setReferencesVisible(true)}
                  icon={
                    <Badge
                      count={_.size(partReference?.references)}
                      size="small"
                      style={{ marginTop: -2 }}
                    />
                  }
                />
              </Tooltip>
            )}
            {isHovering && partReference ? (
              <Space>
                <Button
                  icon={
                    <LikeFilled
                      style={{
                        color:
                          partReference?.feedback?.type ===
                          FeedbackTypeEnum.good
                            ? token.colorSuccess
                            : token.colorTextDisabled,
                      }}
                    />
                  }
                  type="text"
                  onClick={() => handleFeedback(FeedbackTypeEnum.good)}
                />
                <Button
                  icon={
                    <DislikeFilled
                      style={{
                        color:
                          partReference?.feedback?.type === FeedbackTypeEnum.bad
                            ? token.colorError
                            : token.colorTextDisabled,
                      }}
                    />
                  }
                  type="text"
                  onClick={() => handleFeedback(FeedbackTypeEnum.bad)}
                />
              </Space>
            ) : null}
          </StyledMessageInfo>
        </StyledMessageBody>
        {!isAi && (
          <StyledMessageAvatar
            size="large"
            token={token}
            icon={<UserOutlined style={{ fontSize: 14 }} />}
          />
        )}
      </StyledMessage>
      <Drawer
        title={formatMessage({ id: 'text.references' })}
        open={referencesVisible}
        size="large"
        onClose={() => setReferencesVisible(false)}
      >
        <Collapse
          bordered={false}
          defaultActiveKey={['0']}
          expandIcon={({ isActive }) => (
            <CaretRightOutlined rotate={isActive ? 90 : 0} />
          )}
          style={{ background: token.colorBgContainer }}
          items={getReferences()}
        />
      </Drawer>
      <Modal
        title={formatMessage({ id: 'feedback.bad.title' })}
        open={feedbackModalVisible}
        onOk={handleFeedbackSubmit}
        onCancel={() => {
          setFeedbackModalVisible(false);
          setFeedbackTag(undefined);
          setFeedbackMessage(undefined);
        }}
        okButtonProps={{
          disabled: !feedbackTag,
        }}
      >
        <div style={{ marginBottom: 16 }}>
          <Typography.Text type="secondary">
            {formatMessage({ id: 'feedback.bad.description' })}
          </Typography.Text>
        </div>
        <Radio.Group
          value={feedbackTag}
          onChange={(e) => setFeedbackTag(e.target.value)}
          style={{ width: '100%' }}
        >
          <Space direction="vertical" style={{ width: '100%' }}>
            {Object.values(FeedbackTagEnum).map((tag) => (
              <Radio key={tag} value={tag}>
                {formatMessage({ id: `feedback.tag.${tag.toLowerCase()}` })}
              </Radio>
            ))}
          </Space>
        </Radio.Group>
        <Input.TextArea
          value={feedbackMessage}
          onChange={(e) => setFeedbackMessage(e.target.value)}
          placeholder={formatMessage({ id: 'feedback.message.placeholder' })}
          style={{ marginTop: 16 }}
          rows={4}
        />
      </Modal>
    </>
  );
};
