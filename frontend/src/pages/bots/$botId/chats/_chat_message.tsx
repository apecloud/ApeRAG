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
import { useRef, useState } from 'react';
import { BsRobot } from 'react-icons/bs';
import { css, styled, useIntl } from 'umi';

export const StyledMessage = styled('div').withConfig({
  shouldForwardProp: (prop) => !['message'].includes(prop),
})<{ message: ChatMessage }>`
  ${({ message }) => {
    return css`
      display: flex;
      flex-direction: row;
      justify-content: ${message.role === 'human' ? 'flex-end' : 'flex-start'};
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
  shouldForwardProp: (prop) => !['message', 'token'].includes(prop),
})<{ message: ChatMessage; token: GlobalToken }>`
  ${({ message, token }) => {
    return css`
      background: ${message.role === 'human'
        ? token.colorPrimary
        : token.colorBgContainer};
      color: ${message.role === 'human' ? token.colorWhite : token.colorText};
      min-height: 54px;
      padding: 16px;
      border-radius: 8px;
      box-shadow: ${token.boxShadow};
    `;
  }}
`;

export const StyledMessageInfo = styled('div').withConfig({
  shouldForwardProp: (prop) => !['token', 'message'].includes(prop),
})<{ token: GlobalToken; message: ChatMessage }>`
  ${({ token, message }) => {
    return css`
      color: ${token.colorTextSecondary};
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      gap: 16px;
      align-items: center;
      min-height: 32px;
      justify-content: ${message.role === 'human' ? 'flex-end' : 'flex-start'};
    `;
  }}
`;

export const ChatMessageItem = ({
  item,
  loading,
  onVote,
}: {
  item: ChatMessage;
  loading: boolean;
  onVote: (item: ChatMessage, feedback: Feedback) => void;
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

  const getReferences: () => CollapseProps['items'] = () =>
    item.references?.map((reference, index) => {
      if (reference.image_uri) {
        return {
          key: index,
          label: (
            <Typography.Text
              style={{ maxWidth: 400, color: token.colorPrimary }}
              ellipsis
            >
              {index + 1}. {reference.metadata?.name || 'Image Reference'}
            </Typography.Text>
          ),
          children: (
            <div
              style={{
                borderTop: `1px solid ${token.colorBorderSecondary}`,
                paddingTop: 16,
              }}
            >
              <img
                src={reference.image_uri}
                alt="Reference"
                style={{ maxWidth: '100%' }}
              />
              {reference.text && (
                <div style={{ marginTop: 16 }}>
                  <ApeMarkdown>{reference.text}</ApeMarkdown>
                </div>
              )}
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
        };
      }
      return {
        key: index,
        label: (
          <Typography.Text
            style={{ maxWidth: 400, color: token.colorPrimary }}
            ellipsis
          >
            {index + 1}.{' '}
            {reference.metadata?.name || reference.metadata?.source}
          </Typography.Text>
        ),
        children: (
          <div
            style={{
              borderTop: `1px solid ${token.colorBorderSecondary}`,
              paddingTop: 16,
            }}
          >
            <ApeMarkdown>{reference.text || ''}</ApeMarkdown>
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
      };
    });

  const handleFeedback = (type: FeedbackTypeEnum) => {
    if (type === item.feedback?.type) {
      onVote(item, {});
    } else if (type === FeedbackTypeEnum.good) {
      onVote(item, { type });
    } else {
      setFeedbackModalVisible(true);
    }
  };

  const handleFeedbackSubmit = () => {
    onVote(item, {
      type: FeedbackTypeEnum.bad,
      tag: feedbackTag,
      message: feedbackMessage,
    });
    setFeedbackModalVisible(false);
    setFeedbackTag(undefined);
    setFeedbackMessage(undefined);
  };

  return (
    <>
      <StyledMessage message={item}>
        {item.role === 'ai' && (
          <StyledMessageAvatar src={<BsRobot />} size="large" token={token} />
        )}
        <StyledMessageBody ref={hoverRef}>
          <StyledMessageContent token={token} message={item}>
            {item.role === 'human' ? (
              item.data
            ) : loading ? (
              <TypingAnimate />
            ) : (
              <ApeMarkdown>{item.data}</ApeMarkdown>
            )}
          </StyledMessageContent>
          <StyledMessageInfo token={token} message={item}>
            {moment(item.timestamp).format(DATETIME_FORMAT)}
            {!_.isEmpty(item.references) && item.role === 'ai' && (
              <Tooltip title={formatMessage({ id: 'text.references' })}>
                <Button
                  type="text"
                  onClick={() => setReferencesVisible(true)}
                  icon={
                    <Badge
                      count={_.size(item.references)}
                      size="small"
                      style={{ marginTop: -2 }}
                    />
                  }
                />
              </Tooltip>
            )}
            {isHovering && item.role === 'ai' ? (
              <Space>
                <Button
                  icon={
                    <LikeFilled
                      style={{
                        color:
                          item.feedback?.type === FeedbackTypeEnum.good
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
                          item.feedback?.type === FeedbackTypeEnum.bad
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
        {item.role === 'human' && (
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
