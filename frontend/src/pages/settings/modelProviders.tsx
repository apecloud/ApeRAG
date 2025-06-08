import { ModelServiceProvider } from '@/api';
import { PageContainer, PageHeader } from '@/components';
import { MODEL_PROVIDER_ICON } from '@/constants';
import { api } from '@/services';
import { EditOutlined } from '@ant-design/icons';
import {
  Avatar,
  Button,
  Divider,
  Form,
  Input,
  Modal,
  Space,
  Switch,
  Table,
  TableProps,
  Tooltip,
  Typography,
} from 'antd';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { BsRobot } from 'react-icons/bs';

import { useIntl, useModel } from 'umi';

type ListModelProvider = ModelServiceProvider & {
  enabled: boolean;
};

export default () => {
  const { loading, setLoading } = useModel('global');
  const { formatMessage } = useIntl();

  const [supportedModelProviders, setSupportedModelProviders] =
    useState<ModelServiceProvider[]>();

  const [modelProviders, setModelProviders] =
    useState<ModelServiceProvider[]>();

  const [form] = Form.useForm<ListModelProvider>();

  const [visible, setVisible] = useState<boolean>(false);

  const [modal, contextHolder] = Modal.useModal();

  const getSupportedModelProviders = useCallback(async () => {
    setLoading(true);
    const res = await api.supportedModelServiceProvidersGet();
    setSupportedModelProviders(res.data.items);
    setLoading(false);
  }, []);

  const getModelProviders = useCallback(async () => {
    setLoading(true);
    const res = await api.modelServiceProvidersGet();
    setModelProviders(res.data.items);
    setLoading(false);
  }, []);

  const onEditProvider = useCallback(async (item: ListModelProvider) => {
    setVisible(true);
    form.setFieldsValue(item);
  }, []);

  const onUpdateProvider = useCallback(async () => {
    setLoading(true);
    const provider = await form.validateFields();
    if (!provider.name) return;
    await api.modelServiceProvidersProviderPut({
      provider: provider.name,
      modelServiceProviderUpdate: provider,
    });
    await getModelProviders();
    setVisible(false);
    setLoading(false);
  }, []);

  const onToggleProvider = useCallback(
    async (enable: boolean, item: ListModelProvider) => {
      if (!item.name) return;

      if (enable) {
        // add model service provider
        await onEditProvider(item);
      } else {
        // delete model service provider
        const confirmed = await modal.confirm({
          title: formatMessage({ id: 'action.confirm' }),
          content: formatMessage(
            { id: 'model.provider.disable.confirm' },
            { label: item.label },
          ),
          okButtonProps: {
            danger: true,
          },
        });
        if (confirmed) {
          setLoading(true);
          await api.modelServiceProvidersProviderDelete({
            provider: item.name,
          });
          await getModelProviders();
          setLoading(false);
        }
      }
    },
    [onEditProvider, getModelProviders],
  );

  const columns: TableProps<ListModelProvider>['columns'] = useMemo(
    () => [
      {
        title: formatMessage({ id: 'model.provider' }),
        dataIndex: 'label',
        render: (value, record) => {
          return (
            <Space>
              <Avatar
                shape="square"
                src={MODEL_PROVIDER_ICON[record.name || '']}
                icon={<BsRobot />}
              />
              <Typography.Text type={record.enabled ? undefined : 'secondary'}>
                {value}
              </Typography.Text>
            </Space>
          );
        },
      },

      {
        title: formatMessage({ id: 'model.provider.api_key' }),
        dataIndex: 'api_key',
        render: (value, record) => (
          <Typography.Text type={record.enabled ? undefined : 'secondary'}>
            {value ? '************' : ''}
          </Typography.Text>
        ),
      },
      {
        title: formatMessage({ id: 'action.name' }),
        width: 140,
        render: (value, record) => {
          return (
            <Space split={<Divider type="vertical" />}>
              <Tooltip
                title={formatMessage({
                  id: record.enabled
                    ? 'model.provider.disable'
                    : 'model.provider.enable',
                })}
              >
                <Switch
                  size="small"
                  checked={record.enabled}
                  onChange={(v) => onToggleProvider(v, record)}
                />
              </Tooltip>
              <Button
                disabled={!record.enabled}
                type="text"
                icon={<EditOutlined />}
                onClick={() => onEditProvider(record)}
              />
            </Space>
          );
        },
      },
    ],
    [onToggleProvider, onEditProvider],
  );

  const listModelProviders: ListModelProvider[] = useMemo(
    () =>
      supportedModelProviders?.map((smp) => {
        const enabledProvider = modelProviders?.find(
          (mp) => mp.name === smp.name,
        );
        return {
          name: smp.name,
          enabled: enabledProvider !== undefined,
          label: enabledProvider?.label || smp.label,
          api_key: enabledProvider?.api_key,
        };
      }) || [],
    [supportedModelProviders, modelProviders],
  );

  useEffect(() => {
    getSupportedModelProviders();
    getModelProviders();
  }, []);

  return (
    <PageContainer>
      {contextHolder}
      <PageHeader
        title={formatMessage({ id: 'model.provider' })}
        description={formatMessage({ id: 'model.provider.description' })}
      ></PageHeader>
      <Table
        dataSource={listModelProviders}
        bordered
        rowKey="name"
        columns={columns}
        loading={loading}
        pagination={false}
      />
      <Modal
        title={formatMessage({ id: 'model.provider.settings' })}
        onCancel={() => setVisible(false)}
        onOk={onUpdateProvider}
        open={visible}
        width={580}
        okButtonProps={{
          loading,
        }}
      >
        <Divider />
        <Form autoComplete="off" layout="vertical" form={form}>
          <Form.Item
            name="name"
            hidden
            label={formatMessage({ id: 'model.provider' })}
            rules={[
              {
                required: true,
                message: formatMessage({ id: 'model.provider.required' }),
              },
            ]}
          >
            <Input />
          </Form.Item>

          <Form.Item
            name="api_key"
            label={formatMessage({ id: 'model.provider.api_key' })}
            rules={[
              {
                required: true,
                message: formatMessage({
                  id: 'model.provider.api_key.required',
                }),
              },
            ]}
          >
            <Input
              placeholder={formatMessage({ id: 'model.provider.api_key' })}
            />
          </Form.Item>
        </Form>
      </Modal>
    </PageContainer>
  );
};
