import { PageContainer, PageHeader } from '@/components';
import { QuestionCircleOutlined } from '@ant-design/icons';
import {
  Button,
  Card,
  Form,
  Input,
  Space,
  Switch,
  Tooltip,
  Typography,
  message,
} from 'antd';
import { useEffect, useState } from 'react';
import { useIntl } from 'umi';
import { Settings } from '@/api';
import { api } from '@/services';

export default () => {
  const { formatMessage } = useIntl();
  const [form] = Form.useForm();
  const [isTesting, setIsTesting] = useState(false);
  const [testResult, setTestResult] = useState<{
    type: 'success' | 'danger' | 'warning';
    message: string;
  } | null>(null);

  const useMineru = Form.useWatch('use_mineru', form);
  const mineruApiToken = Form.useWatch('mineru_api_token', form);

  const onFinish = async (values: Settings) => {
    try {
      await api.settingsPut({ settings: values });
      message.success(formatMessage({ id: 'tips.update.success' }));
    } catch (error) {
      message.error(formatMessage({ id: 'tips.update.error' }));
    }
  };

  const handleTestMineruToken = async (tokenToTest?: string) => {
    setIsTesting(true);
    setTestResult(null);
    const token = tokenToTest ?? form.getFieldValue('mineru_api_token');

    if (!token && !tokenToTest) {
      message.error(
        formatMessage({ id: 'settings.mineru_api_token.required' }),
      );
      setIsTesting(false);
      return;
    }

    try {
      const response = await api.settingsTestMineruTokenPost({
        settingsTestMineruTokenPostRequest: { token },
      });
      const result = response.data;

      if (response.status === 404) {
        setTestResult({
          type: 'warning',
          message: formatMessage({
            id: 'settings.mineru_api_token.test.not_set',
          }),
        });
        return;
      }

      const { status_code, data: res } = result as any;

      if (status_code === 401) {
        if (res && res.msgCode === 'A0211') {
          setTestResult({
            type: 'danger',
            message: formatMessage({
              id: 'settings.mineru_api_token.test.expired',
            }),
          });
        } else {
          setTestResult({
            type: 'danger',
            message: formatMessage({
              id: 'settings.mineru_api_token.test.invalid',
            }),
          });
        }
      } else if (status_code === 404 || (res && res.code === -60012)) {
        setTestResult({
          type: 'success',
          message: formatMessage({
            id: 'settings.mineru_api_token.test.valid',
          }),
        });
      } else {
        setTestResult({
          type: 'danger',
          message: `${formatMessage({
            id: 'settings.mineru_api_token.test.error',
          })}: ${res?.msg || 'Unknown error'}`,
        });
      }
    } catch (error) {
      setTestResult({
        type: 'danger',
        message: formatMessage({
          id: 'settings.mineru_api_token.test.fetch_error',
        }),
      });
    } finally {
      setIsTesting(false);
    }
  };

  const fetchInitialState = async () => {
    try {
      const response = await api.settingsGet();
      form.setFieldsValue(response.data);
      if (response.data.mineru_api_token) {
        handleTestMineruToken(response.data.mineru_api_token);
      }
    } catch (error) {
      // do nothing
    }
  };

  useEffect(() => {
    fetchInitialState();
  }, []);

  return (
    <PageContainer>
      <PageHeader title={formatMessage({ id: 'settings.miscellaneous' })} />
      <Card>
        <Form form={form} layout="vertical" onFinish={onFinish}>
          <Form.Item
            label={
              <Space>
                {formatMessage({ id: 'settings.use_mineru' })}
                <Tooltip
                  title={formatMessage({ id: 'settings.use_mineru.tooltip' })}
                >
                  <QuestionCircleOutlined />
                </Tooltip>
              </Space>
            }
            name="use_mineru"
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>

          {useMineru && (
            <>
              <Form.Item
                label="MinerU API Token"
                extra={
                  <>
                    <div>
                      {formatMessage({ id: 'settings.mineru_api_token.extra' })}
                    </div>
                    {testResult && (
                      <Typography.Text type={testResult.type}>
                        {testResult.message}
                      </Typography.Text>
                    )}
                  </>
                }
              >
                <Space.Compact style={{ width: '100%' }}>
                  <Form.Item
                    name="mineru_api_token"
                    noStyle
                    rules={[
                      {
                        required: true,
                        message: formatMessage({
                          id: 'settings.mineru_api_token.required',
                        }),
                      },
                    ]}
                  >
                    <Input.Password
                      placeholder={formatMessage({
                        id: 'settings.mineru_api_token.placeholder.new',
                      })}
                      autoComplete="off"
                      onChange={() => setTestResult(null)}
                    />
                  </Form.Item>
                  <Button
                    type="primary"
                    loading={isTesting}
                    disabled={!mineruApiToken}
                    onClick={() => handleTestMineruToken()}
                  >
                    {formatMessage({
                      id: 'settings.mineru_api_token.test.btn',
                    })}
                  </Button>
                </Space.Compact>
              </Form.Item>
            </>
          )}
          <Form.Item>
            <Button type="primary" htmlType="submit">
              {formatMessage({ id: 'action.save' })}
            </Button>
          </Form.Item>
        </Form>
      </Card>
    </PageContainer>
  );
};
