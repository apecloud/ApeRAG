import { NAVIGATION_WIDTH, SIDEBAR_WIDTH } from '@/constants';
import { PauseCircleFilled, PlayCircleFilled } from '@ant-design/icons';
import { GlobalToken, Input, Space, theme } from 'antd';
import alpha from 'color-alpha';
import _ from 'lodash';
import { useState } from 'react';
import { css, styled, useIntl } from 'umi';

export const StyledChatInputContainer = styled('div').withConfig({
  shouldForwardProp: (prop) => !['token'].includes(prop),
})<{ token: GlobalToken }>`
  ${({ token }) => {
    return css`
      padding: 24px;
      position: fixed;
      left: ${SIDEBAR_WIDTH + NAVIGATION_WIDTH}px;
      bottom: 0;
      right: 0;
      backdrop-filter: blur(20px);
      background-color: ${alpha(token.colorBgLayout, 0.85)};
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 12px;
    `;
  }}
`;

export const StyledChatInput = styled(Input.TextArea).withConfig({
  shouldForwardProp: (prop) => !['token'].includes(prop),
})<{ token: GlobalToken }>`
  ${() => {
    return css`
      background: none !important;
      border: none !important;
      box-shadow: none !important;
      resize: none !important;
      padding: 0;
    `;
  }}
`;

export const StyledChatInputOuter = styled('div').withConfig({
  shouldForwardProp: (prop) => !['token'].includes(prop),
})<{ token: GlobalToken }>`
  ${({ token }) => {
    return css`
      width: 75%;
      box-shadow: ${token.boxShadow};
      border-radius: 12px;
      background: ${token.colorBgContainer};
      padding: 12px;
    `;
  }}
`;

export const ChatInput = ({
  onCancel,
  onSubmit,
  loading,
  disabled,
}: {
  disabled: boolean;
  onCancel: () => void;
  onSubmit: (v: string) => void;
  loading?: boolean;
}) => {
  const { token } = theme.useToken();
  const [value, setValue] = useState<string>();
  const { formatMessage } = useIntl();

  const onPressEnter = async () => {
    if (disabled) return;

    const data = _.trim(value);
    if (loading || _.isEmpty(data)) return;
    onSubmit(data);
    setValue(undefined);
  };

  return (
    <StyledChatInputContainer token={token}>
      <StyledChatInputOuter token={token}>
        <StyledChatInput
          value={value}
          onChange={(e) => setValue(e.currentTarget.value)}
          token={token}
          onKeyDown={(e) => {
            if (e.key !== 'Enter' || e.shiftKey) return;
            onPressEnter();
            e.preventDefault();
          }}
          placeholder={formatMessage(
            { id: 'chat.input_placeholder' },
            { title: APERAG_CONFIG.title },
          )}
          rows={2}
          autoSize={false}
          autoFocus
        />
        <Space style={{ display: 'flex', justifyContent: 'space-between' }}>
          <div />
          {/* <Button
            type="primary"
            loading={loading}
            shape="circle"
            onClick={() => onPressEnter()}
            icon={<CaretRightFilled />}
          /> */}

          <div
            onClick={() => {
              if (disabled) return;

              if (loading) {
                onCancel();
              } else {
                onPressEnter();
              }
            }}
            style={{
              cursor: 'pointer',
              color: disabled ? token.colorTextDisabled : token.colorPrimary,
              fontSize: 32,
              width: 32,
              height: 32,
              display: 'flex',
            }}
          >
            {loading ? <PauseCircleFilled /> : <PlayCircleFilled />}
          </div>
        </Space>
      </StyledChatInputOuter>
    </StyledChatInputContainer>
  );
};
