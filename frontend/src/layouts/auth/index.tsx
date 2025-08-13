import { Result, theme } from 'antd';
import { AuthProvider as OIDCAuthProvier } from 'oidc-react';
import { UndrawSecureServer } from 'react-undraw-illustrations';
import { FormattedMessage } from 'umi';
import AuthCookie from './cookie';
import AuthNone from './none';
import AuthOidc from './oidc';

export const AuthProvider = (props: { children: JSX.Element }) => {
  const { token } = theme.useToken();
  const authConfig = APERAG_CONFIG.auth || {};

  const type = authConfig.type;

  if (type === 'none' || type === 'cookie') return props.children;

  if (type === 'auth0' || type === 'authing' || type === 'logto') {
    const auth_domain = authConfig[type]?.auth_domain;
    const auth_app_id = authConfig[type]?.auth_app_id;
    return (
      <OIDCAuthProvier
        authority={auth_domain}
        clientId={auth_app_id}
        responseType="code"
        scope="openid email profile"
        autoSignIn={false}
      >
        {props.children}
      </OIDCAuthProvier>
    );
  }

  return (
    <>
      <Result
        style={{ paddingTop: 120 }}
        icon={
          <UndrawSecureServer
            primaryColor={token.colorPrimary}
            height="200px"
          />
        }
        title={<FormattedMessage id="text.authorize.error" />}
        subTitle={<FormattedMessage id="text.authorize.error.description" />}
      />
    </>
  );
};

export const Auth = ({ children }: { children: React.ReactNode }) => {
  const { type } = APERAG_CONFIG.auth || {};

  if (type === 'auth0' || type === 'authing' || type === 'logto') {
    return <AuthOidc>{children}</AuthOidc>;
  }

  if (type === 'none') {
    return <AuthNone>{children}</AuthNone>;
  }

  if (type === 'cookie') {
    return <AuthCookie>{children}</AuthCookie>;
  }

  // 默认情况下，如果没有配置认证类型或者类型为空，使用 none 认证
  return <AuthNone>{children}</AuthNone>;
};
