import { Collection, CollectionView } from '@/api';
import { ModelSelect } from '@/components';
import { NAVIGATION_WIDTH, SIDEBAR_WIDTH } from '@/constants';
import { CaretRightOutlined, PauseOutlined } from '@ant-design/icons';
import { useLocalStorageState } from 'ahooks';
import {
  Button,
  GlobalToken,
  Mentions,
  Space,
  Tag,
  theme,
  Tooltip,
} from 'antd';
import alpha from 'color-alpha';
import _ from 'lodash';
import { useCallback, useEffect, useState } from 'react';
import { toast } from 'react-toastify';
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
  const [selectedCollections, setSelectedCollections] = useState<
    CollectionView[]
  >([]);
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

  const handleCollectionRemove = useCallback((collection: CollectionView) => {
    setSelectedCollections((cs) => cs.filter((c) => c.id !== collection.id));
  }, []);

  const onPressEnter = async () => {
    if (disabled || isComposing) return;
    
    const { provider, model } = getProviderByModelName(modelName, 'completion');
    if(!model) {
      toast.error(formatMessage({ id: 'default.models.select.placeholder' }));
      return;
    }

    const _query = _.trim(query);
    if (loading || _.isEmpty(_query)) return;

    const data = {
      query: _query,
    };



    Object.assign(data, {
      collections: selectedCollections,
      completion: {
        model: model?.model,
        model_service_provider: provider?.name,
        custom_llm_provider: model?.custom_llm_provider,
      },
      web_search_enabled: webSearchEnabled,
      language: getLocale(),
    });
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
          placeholder="Enter @ to select the collection and send a message to ApeRAG."
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
            {bot?.type === 'agent' &&
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
              ))}
          </div>
          <Space style={{ display: 'flex', gap: 12 }}>
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
                >
                  <svg
                    viewBox="0 0 1024 1024"
                    version="1.1"
                    xmlns="http://www.w3.org/2000/svg"
                    width="16"
                    height="16"
                  >
                    <path
                      fill={
                        webSearchEnabled ? token.colorPrimary : token.colorText
                      }
                      d="M515.84 834.56c-1.28 2.56-2.56 5.546667-3.84 8.106667-5.546667-10.24-10.24-21.333333-15.36-32a253.909333 253.909333 0 0 1-47.36-128.426667c-10.666667-41.386667-18.346667-84.053333-21.333333-128h43.52c14.933333-32.853333 36.693333-61.866667 63.573333-85.333333h-107.093333c5.973333-101.12 34.986667-199.68 84.053333-288 43.093333 77.653333 69.12 162.56 79.786667 250.453333 25.6-12.8 53.76-21.333333 83.626666-24.32-10.666667-76.8-32.426667-151.893333-65.28-222.293333 63.573333 18.773333 119.893333 55.893333 162.56 106.666666 38.4 45.226667 63.146667 99.413333 74.24 157.44 38.826667 26.026667 69.973333 62.72 90.026667 105.813334 1.28-14.08 2.133333-28.16 2.133333-42.666667 0-235.52-191.146667-426.666667-426.666666-426.666667S85.333333 276.48 85.333333 512s191.146667 426.666667 426.666667 426.666667c51.2 0 99.84-9.386667 145.066667-26.026667a255.573333 255.573333 0 0 1-141.226667-78.08zM251.733333 732.586667c-42.666667-50.346667-69.973333-112.213333-78.08-177.92h168.96c5.12 98.56 29.866667 194.986667 71.68 284.586666a336.853333 336.853333 0 0 1-162.56-106.666666zM342.613333 469.333333H173.653333c8.106667-65.706667 35.413333-127.573333 78.08-177.92 42.666667-50.346667 99.413333-87.466667 162.56-106.666666C372.053333 273.92 347.733333 370.773333 342.613333 469.333333z"
                    ></path>
                    <path
                      fill={
                        webSearchEnabled ? token.colorPrimary : token.colorText
                      }
                      d="M990.293333 908.373333l-132.693333-132.693333c24.746667-32.853333 38.826667-72.533333 38.826667-114.346667 0-51.2-20.053333-99.413333-56.32-135.68A190.762667 190.762667 0 0 0 704.426667 469.333333c-51.2 0-99.413333 20.053333-135.68 56.32-36.266667 36.266667-56.32 84.48-56.32 135.68 0 51.2 20.053333 99.413333 56.32 135.68 36.266667 36.266667 84.48 56.32 135.68 56.32 31.146667 0 61.013333-8.106667 87.893333-22.186666l137.813333 137.813333c8.533333 8.533333 19.2 12.373333 30.293334 12.373333 11.093333 0 21.76-4.266667 30.293333-12.373333a42.496 42.496 0 0 0 0-60.16l-0.426667-0.426667z m-361.813333-171.52c-20.053333-20.053333-31.146667-46.933333-31.146667-75.52 0-28.586667 11.093333-55.04 31.146667-75.52s46.933333-31.146667 75.52-31.146666c28.586667 0 55.04 11.093333 75.52 31.146666s31.146667 46.933333 31.146667 75.52c0 28.586667-11.093333 55.04-31.146667 75.52-40.533333 40.533333-110.506667 40.533333-151.04 0z"
                    ></path>
                  </svg>
                  Search
                </Button>
              </Tooltip>
            )}

            {bot?.type === 'agent' && (
              <Tooltip title={modelName}>
                <ModelSelect
                  model="completion"
                  showSearch
                  style={{ width: 220 }}
                  value={modelName}
                  onChange={setModelName}
                  placeholder={formatMessage({ id: 'default.models.select.placeholder' })}
                  tagfilters={[
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
