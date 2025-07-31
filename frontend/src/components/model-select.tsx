import { MODEL_PROVIDER_ICON } from '@/constants';
import { Avatar, Select, SelectProps, Space, theme, Typography } from 'antd';
import _ from 'lodash';
import { useMemo } from 'react';
import { UndrawEmpty } from 'react-undraw-illustrations';
import { FormattedMessage, useModel } from 'umi';

type ModelSelectProps = {
  model: 'embedding' | 'completion' | 'rerank';
  tagFilters?: any[]; // Optional tag filters for this model select
};

export const ModelSelect = (props: SelectProps & ModelSelectProps) => {
  const { token } = theme.useToken();
  const { availableModels, getProviderByModelName } = useModel('models');

  // Filter models based on tagFilters if provided
  const filterModelsByTags = (models: any[], tagFilters?: any[]) => {
    if (!tagFilters || tagFilters.length === 0) {
      return models;
    }

    return models.filter((model) => {
      if (!model.tags || model.tags.length === 0) {
        return false;
      }

      // Apply tag filters logic
      return tagFilters.some((filter) => {
        const { operation, tags } = filter;
        if (operation === 'OR') {
          return tags.some((tag: string) => model.tags.includes(tag));
        } else if (operation === 'AND') {
          return tags.every((tag: string) => model.tags.includes(tag));
        }
        return false;
      });
    });
  };

  const modelOptions = useMemo(
    () =>
      _.map(availableModels, (provider) => {
        const models = provider[props.model] || [];
        const filteredModels = filterModelsByTags(models, props.tagFilters);

        return {
          label: (
            <Space>
              <Avatar
                size={24}
                shape="square"
                src={
                  provider.name ? MODEL_PROVIDER_ICON[provider.name] : undefined
                }
              />
              <span>{provider.label || provider.name}</span>
            </Space>
          ),
          options: filteredModels.map((item) => {
            return {
              label: item.model,
              value: item.model,
            };
          }),
        };
      }).filter((item) => !_.isEmpty(item.options)),
    [availableModels, props.model, props.tagFilters],
  );

  const { provider } = getProviderByModelName(props.value, props.model);

  return (
    <Select
      {...props}
      options={modelOptions}
      suffixIcon={null}
      labelRender={({ label, value }) => {
        return (
          <Space style={{ alignItems: 'center' }}>
            {provider?.name && (
              <Avatar
                size={24}
                shape="square"
                src={MODEL_PROVIDER_ICON[provider.name]}
                style={{
                  transform: 'translateY(-1px)',
                }}
              />
            )}
            {label || value}
          </Space>
        );
      }}
      notFoundContent={
        <Space
          direction="vertical"
          style={{
            display: 'flex',
            alignItems: 'center',
            marginBlock: 24,
          }}
        >
          <UndrawEmpty primaryColor={token.colorPrimary} height="80px" />
          <Typography.Text type="secondary">
            <FormattedMessage id="collection.model_not_found" />
          </Typography.Text>
        </Space>
      }
    />
  );
};
