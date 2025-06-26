import { Login } from '@/api';
import bgDark from '@/assets/page/signin-dark.svg';
import bgLight from '@/assets/page/signin-light.svg';
import { PageContainer } from '@/components';
import { api } from '@/services';
import { KeyOutlined, UserOutlined } from '@ant-design/icons';
import { Button, Card, Divider, Form, Input, Space, Typography } from 'antd';
import { useCallback } from 'react';
import { toast } from 'react-toastify';
import {
  FormattedMessage,
  Link,
  useIntl,
  useModel,
  useSearchParams,
} from 'umi';

export default () => {
  const [form] = Form.useForm<Login>();
  const { formatMessage } = useIntl();
  const { themeName, loading, setLoading } = useModel('global');
  const [searchParams] = useSearchParams();
  const redirectUri = searchParams.get('redirectUri');
  const redirectString = redirectUri
    ? '?redirectUri=' + encodeURIComponent(redirectUri)
    : '';

  const onFinish = useCallback(async () => {
    const values = await form.validateFields();
    setLoading(true);

    api
      .loginPost({ login: values })
      .then(() => {
        setLoading(false);
        toast.success(formatMessage({ id: 'user.signin_success' }));
        window.location.href = redirectUri || '/';
      })
      .catch(() => {
        setLoading(false);
      });
  }, []);

  return (
    <PageContainer
      height="fixed"
      width="auto"
      style={{
        backgroundImage: `url(${themeName === 'dark' ? bgDark : bgLight})`,
        backgroundPosition: 'top center',
        backgroundRepeat: 'no-repeat',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexFlow: 'wrap',
      }}
    >
      <Card variant="borderless" style={{ width: 400 }}>
        <Space style={{ display: 'flex', justifyContent: 'space-between' }}>
          <Typography.Title level={3} style={{ margin: 0 }}>
            <FormattedMessage id="user.signin" />
          </Typography.Title>
          <Link to={`/accounts/reset${redirectString}`}>
            <FormattedMessage id="user.forget_password" />
          </Link>
        </Space>
        <Divider />
        <Form
          layout="vertical"
          size="large"
          form={form}
          onFinish={onFinish}
          autoComplete="off"
        >
          <Form.Item
            required
            name="username"
            label={formatMessage({ id: 'user.username' })}
            rules={[
              {
                required: true,
                message: formatMessage({ id: 'user.username_required' }),
              },
            ]}
          >
            <Input
              prefix={
                <Typography.Text type="secondary">
                  <UserOutlined />
                </Typography.Text>
              }
              style={{ fontSize: 'inherit' }}
              placeholder={formatMessage({ id: 'user.username' })}
            />
          </Form.Item>
          <Form.Item
            required
            name="password"
            label={formatMessage({ id: 'user.password' })}
            rules={[
              {
                required: true,
                message: formatMessage({ id: 'user.password_required' }),
              },
            ]}
          >
            <Input.Password
              style={{ fontSize: 'inherit' }}
              prefix={
                <Typography.Text type="secondary">
                  <KeyOutlined />
                </Typography.Text>
              }
              placeholder={formatMessage({ id: 'user.password' })}
            />
          </Form.Item>

          <Button loading={loading} htmlType="submit" block type="primary">
            <FormattedMessage id="user.signin" />
          </Button>
        </Form>

        <Divider />
        <Space
          style={{
            display: 'flex',
            justifyContent: 'center',
            marginTop: 4,
          }}
        >
          <Typography.Text type="secondary">
            <FormattedMessage id="user.not_have_account" />
          </Typography.Text>

          <Link to={`/accounts/signup${redirectString}`}>
            <FormattedMessage id="user.signup" />
          </Link>
        </Space>
      </Card>
    </PageContainer>
  );
};
