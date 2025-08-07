import { Login } from '@/api';
import bgDark from '@/assets/page/signin-dark.svg';
import bgLight from '@/assets/page/signin-light.svg';
import { PageContainer } from '@/components';
import { api } from '@/services';
import { KeyOutlined, UserOutlined } from '@ant-design/icons';
import { Alert, Button, Card, Divider, Form, Input, Space, Typography } from 'antd';
import { useCallback, useEffect, useState } from 'react';
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
  const [loginMethods, setLoginMethods] = useState<string[]>([]);
  
  // Check for OAuth error parameters
  const oauthError = searchParams.get('error');

  useEffect(() => {
    // Fetch available login methods from the backend config
    // Use native fetch to avoid axios interceptors that might cause redirects
    fetch('/api/v1/config', {
      credentials: 'include', // Include cookies if any
    })
      .then((res) => {
        if (res.ok) {
          return res.json();
        }
        throw new Error('Failed to fetch config');
      })
      .then((data) => {
        if (data && Array.isArray(data.login_methods)) {
          setLoginMethods(data.login_methods);
        }
      })
      .catch(() => {
        // Fallback to local if API fails
        setLoginMethods(['local']);
      });
  }, []);

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
  }, [redirectUri]);

  const hasSocialLogin =
    loginMethods.includes('google') || loginMethods.includes('github');
  const hasLocalLogin = loginMethods.includes('local');

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
          {hasLocalLogin && (
            <Link to={`/accounts/reset${redirectString}`}>
              <FormattedMessage id="user.forget_password" />
            </Link>
          )}
        </Space>
        <Divider />

        {/* Display OAuth error messages */}
        {oauthError && (
          <>
            <Alert
              message={
                oauthError === 'oauth_failed' 
                  ? formatMessage({ id: 'user.oauth_callback_failed' })
                  : oauthError === 'oauth_invalid'
                  ? formatMessage({ id: 'user.oauth_invalid_params' })
                  : oauthError === 'oauth_unauthorized'
                  ? formatMessage({ id: 'user.oauth_unauthorized' })
                  : oauthError === 'oauth_server_error'
                  ? formatMessage({ id: 'user.oauth_server_error' })
                  : formatMessage({ id: 'user.oauth_error' })
              }
              description={
                oauthError === 'oauth_failed'
                  ? formatMessage({ id: 'user.oauth_callback_failed_desc' })
                  : oauthError === 'oauth_invalid'
                  ? formatMessage({ id: 'user.oauth_invalid_params_desc' })
                  : oauthError === 'oauth_unauthorized'
                  ? formatMessage({ id: 'user.oauth_unauthorized_desc' })
                  : oauthError === 'oauth_server_error'
                  ? formatMessage({ id: 'user.oauth_server_error_desc' })
                  : formatMessage({ id: 'user.oauth_error_desc' })
              }
              type="error"
              showIcon
              style={{ marginBottom: 16 }}
            />
          </>
        )}

        {hasLocalLogin && (
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
        )}

        {hasLocalLogin && hasSocialLogin && (
          <Divider>
            <FormattedMessage id="user.or" />
          </Divider>
        )}

        {hasSocialLogin && (
          <Space direction="vertical" style={{ width: '100%' }}>
            {loginMethods.includes('google') && (
              <Button
                icon={<i className="ri-google-fill" />}
                block
                onClick={async () => {
                  try {
                    // Store provider info for callback page
                    localStorage.setItem('oauth_provider', 'google');
                    console.log('Google OAuth - stored provider in localStorage');
                    
                    const response = await fetch('/api/v1/auth/google/authorize');
                    const data = await response.json();
                    console.log('Google OAuth authorize response:', data);
                    if (data.authorization_url) {
                      console.log('Redirecting to Google OAuth:', data.authorization_url);
                      window.location.href = data.authorization_url;
                    }
                  } catch (error) {
                    console.error('Google OAuth error:', error);
                    toast.error(formatMessage({ id: 'user.oauth_authorize_failed' }));
                  }
                }}
              >
                <FormattedMessage id="user.signin_with_google" />
              </Button>
            )}
            {loginMethods.includes('github') && (
              <Button
                icon={<i className="ri-github-fill" />}
                block
                onClick={async () => {
                  try {
                    // Store provider info for callback page
                    localStorage.setItem('oauth_provider', 'github');
                    console.log('GitHub OAuth - stored provider in localStorage');
                    
                    const response = await fetch('/api/v1/auth/github/authorize');
                    const data = await response.json();
                    console.log('GitHub OAuth authorize response:', data);
                    if (data.authorization_url) {
                      console.log('Redirecting to GitHub OAuth:', data.authorization_url);
                      window.location.href = data.authorization_url;
                    }
                  } catch (error) {
                    console.error('GitHub OAuth error:', error);
                    toast.error(formatMessage({ id: 'user.oauth_authorize_failed' }));
                  }
                }}
              >
                <FormattedMessage id="user.signin_with_github" />
              </Button>
            )}
          </Space>
        )}

        {hasLocalLogin && (
          <>
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
          </>
        )}
      </Card>
    </PageContainer>
  );
};
