import { Collection } from '@/api';
import { ModelSelect } from '@/components';
import { NAVIGATION_WIDTH, SIDEBAR_WIDTH } from '@/constants';
import {
  CaretRightOutlined,
  GlobalOutlined,
  PauseOutlined,
} from '@ant-design/icons';
import { useLocalStorageState } from 'ahooks';
import {
  Button,
  GlobalToken,
  Mentions,
  Space,
  Tag,
  theme,
  Tooltip,
  Typography,
} from 'antd';
import alpha from 'color-alpha';
import _ from 'lodash';
import { useCallback, useEffect, useState } from 'react';
import { css, getLocale, styled, useIntl, useModel } from 'umi';

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

export type ChatInputProps = {
  disabled: boolean;
  onCancel: () => void;
  onSubmit: (params: {
    query: string;
    collections?: Collection[];
    completion?: {
      model?: string;
      model_service_provider?: string;
      custom_llm_provider?: string;
    };
    web_search_enabled?: boolean;
  }) => void;
  loading?: boolean;
};

export const ChatInput = ({
  onCancel,
  onSubmit,
  loading,
  disabled,
}: ChatInputProps) => {
  const { token } = theme.useToken();
  const { collections, getCollections } = useModel('collection');
  const [selectedCollections, setSelectedCollections] = useState<Collection[]>(
    [],
  );
  const { getAvailableModels, getProviderByModelName } = useModel('models');
  const { bot } = useModel('bot');
  const [webSearchEnabled, setWebSearchEnabled] = useState(false);
  const [modelName, setModelName] = useLocalStorageState<string | undefined>(
    'model',
  );

  const [query, setQuery] = useState<string>();
  const [isComposing, setIsComposing] = useState<boolean>(false);
  const { formatMessage } = useIntl();

  const handleCollectionSelect = useCallback(
    (id?: string) => {
      if (!id) return;
      if (selectedCollections.find((item) => item.id === id)) {
        return;
      } else {
        const c = collections?.find((item) => item.id === id);
        if (c) {
          setSelectedCollections((cs) => cs.concat(c));
        }
      }
    },
    [collections, selectedCollections],
  );

  const handleCollectionRemove = useCallback((collection: Collection) => {
    setSelectedCollections((cs) => cs.filter((c) => c.id !== collection.id));
  }, []);

  const onPressEnter = async () => {
    if (disabled || isComposing) return;
    const _query = _.trim(query);
    if (loading || _.isEmpty(_query)) return;

    const data = {
      query: _query,
      language: getLocale(), // 所有Bot类型都添加当前用户语言偏好
    };

    if (bot?.type === 'agent') {
      const { provider, model } = getProviderByModelName(
        modelName,
        'completion',
      );
      Object.assign(data, {
        collections: selectedCollections,
        completion: {
          model: model?.model,
          model_service_provider: provider?.name,
          custom_llm_provider: model?.custom_llm_provider,
        },
        web_search_enabled: webSearchEnabled,
      });
    }
    onSubmit(data);
    setQuery(undefined);
  };

  useEffect(() => {
    getCollections();
    getAvailableModels();
  }, []);

  return (
    <StyledChatInputContainer token={token}>
      <StyledChatInputOuter token={token}>
        <Mentions
          value={query}
          onChange={(v) => setQuery(v)}
          onSelect={(option) => {
            handleCollectionSelect(option.value);
            setQuery(query?.replace(/@$/g, ''));
          }}
          options={collections?.map((c) => ({
            label: c.title,
            value: c.id,
          }))}
          onKeyDown={(e) => {
            if (
              e.key !== 'Enter' ||
              e.shiftKey ||
              e.currentTarget.nextElementSibling
            )
              return;
            onPressEnter();
            e.preventDefault();
          }}
          placeholder={formatMessage(
            { id: 'chat.input_placeholder' },
            { title: APERAG_CONFIG.title },
          )}
          onCompositionStart={() => setIsComposing(true)}
          onCompositionEnd={() => setIsComposing(false)}
          rows={2}
          autoSize={false}
          autoFocus
          variant="borderless"
          prefix={bot?.type !== 'agent' ? 'should_not_support_mentions' : '@'}
        />
        <Space
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            paddingTop: 4,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {bot?.type === 'agent' && (
              <Tooltip title="Web search">
                <Button
                  type="text"
                  onClick={() => setWebSearchEnabled(!webSearchEnabled)}
                  style={{
                    color: webSearchEnabled
                      ? token.colorPrimary
                      : token.colorText,
                  }}
                  shape="circle"
                  icon={<GlobalOutlined />}
                />
              </Tooltip>
            )}
            {bot?.type === 'agent' &&
              (selectedCollections.length ? (
                selectedCollections.map((c) => (
                  <Tag
                    key={c.id}
                    closable
                    onClose={() => handleCollectionRemove(c)}
                    style={{
                      margin: 1,
                    }}
                  >
                    {c.title}
                  </Tag>
                ))
              ) : (
                <Typography.Text type="secondary">
                  Enter the @ symbol to select a collection
                </Typography.Text>
              ))}
          </div>
          <Space style={{ display: 'flex', gap: 12 }}>
            {bot?.type === 'agent' && (
              <Tooltip title={modelName}>
                <ModelSelect
                  model="completion"
                  showSearch
                  style={{ width: 220 }}
                  value={modelName}
                  onChange={setModelName}
                  tagFilters={[
                    {
                      operation: 'OR',
                      tags: ['enable_for_agent'],
                    },
                  ]}
                />
              </Tooltip>
            )}

            <Button
              onClick={() => {
                if (disabled) return;

                if (loading) {
                  onCancel();
                } else {
                  onPressEnter();
                }
              }}
              shape="circle"
              type="primary"
              icon={loading ? <PauseOutlined /> : <CaretRightOutlined />}
            />
          </Space>
        </Space>
      </StyledChatInputOuter>
    </StyledChatInputContainer>
  );
};
