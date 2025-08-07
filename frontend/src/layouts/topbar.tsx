import { LOCALES, TOPBAR_HEIGHT } from '@/constants';
import { api } from '@/services';
import { getLogo } from '@/utils';
import {
  CheckOutlined,
  GithubOutlined,
  LoginOutlined,
  MoonOutlined,
  SunOutlined,
  UserOutlined,
} from '@ant-design/icons';
import {
  Avatar,
  Button,
  ButtonProps,
  Card,
  Divider,
  Dropdown,
  GlobalToken,
  Image,
  Menu,
  MenuProps,
  Space,
  theme,
  Typography,
} from 'antd';
import alpha from 'color-alpha';
import { useAuth } from 'oidc-react';
import { useMemo } from 'react';
import {
  css,
  getLocale,
  Link,
  setLocale,
  styled,
  useIntl,
  useLocation,
  useModel,
} from 'umi';
import styles from './topbar.less';

const StyledTopbar = styled('header').withConfig({
  shouldForwardProp: (prop) => !['token'].includes(prop),
})<{
  token: GlobalToken;
}>`
  ${({ token }) => css`
    padding-inline: 18px;
    height: ${TOPBAR_HEIGHT}px;
    border-bottom: 1px solid ${token.colorBorderSecondary};
    display: flex;
    justify-content: space-between;
    backdrop-filter: blur(20px);
    background-color: ${alpha(token.colorBgLayout, 0.75)};
    z-index: ${token.zIndexPopupBase};
    position: fixed;
    left: 0;
    right: 0;
    top: 0;
    box-shadow: ${token.boxShadowTertiary};
  `}
`;

export default () => {
  const { token } = theme.useToken();
  const { themeName, setThemeName } = useModel('global');
  const { user } = useModel('user');
  const locale = getLocale();
  const location = useLocation();
  const { formatMessage } = useIntl();

  const { type } = APERAG_CONFIG.auth || {};
  const oidc =
    type && ['auth0', 'authing', 'logto'].includes(type)
      ? useAuth()
      : undefined;

  const login = async () => {};
  const logout = async () => {
    if (oidc) {
      await oidc?.signOut();
      window.location.reload();
    }
    if (type === 'cookie') {
      const res = await api.logoutPost();
      if (res.status === 200) window.location.reload();
    }
  };

  const menuItems: MenuProps['items'] = useMemo(() => {
    let data: MenuProps['items'] = [
      ...LOCALES.map(({ label, key }) => ({
        key,
        label,
        icon: (
          <CheckOutlined
            style={{ visibility: locale === key ? 'visible' : 'hidden' }}
          />
        ),
        onClick: () => setLocale(key),
      })),
      { type: 'divider' },
    ];
    data = data.concat({
      key: 'logtout',
      label: user
        ? formatMessage({ id: 'action.signout' })
        : formatMessage({ id: 'action.signin' }),
      icon: <LoginOutlined />,
      onClick: user ? logout : login,
    });
    return data;
  }, [user]);

  const rightButtonItemProps: ButtonProps = {
    type: 'text',
    shape: 'circle',
  };

  return (
    <>
      <StyledTopbar token={token}>
        <Link to="/" className={styles.logo}>
          <Space>
            <Image
              className={styles.img}
              src={getLogo(themeName)}
              preview={false}
            />
            <Typography.Text className={styles.text}>
              {APERAG_CONFIG.title}
            </Typography.Text>
          </Space>
        </Link>
        <Space className={styles.topnav}>
          <Link to="/marketplace">
            <Button type="text">
              {formatMessage({ id: 'collection.marketplace.title' })}
            </Button>
          </Link>
        </Space>
        <Space>
          {APERAG_CONFIG.github && (
            <a href={APERAG_CONFIG.github} target="_blank" rel="noreferrer">
              <Button {...rightButtonItemProps} icon={<GithubOutlined />} />
            </a>
          )}
          <Button
            disabled={location.pathname === '/'}
            onClick={() => {
              setThemeName(themeName === 'dark' ? 'light' : 'dark');
            }}
            {...rightButtonItemProps}
            icon={themeName === 'dark' ? <SunOutlined /> : <MoonOutlined />}
          />
          <Dropdown
            dropdownRender={() => {
              return (
                <Card
                  styles={{ body: { padding: 4 } }}
                  className={styles.userContainer}
                >
                  {user && (
                    <>
                      <Space className={styles.info}>
                        <Avatar
                          size="large"
                          icon={<UserOutlined style={{ fontSize: 14 }} />}
                        />
                        <div>
                          <Typography.Text>{user?.username}</Typography.Text>
                          {user?.email && (
                            <Typography.Text className={styles.email}>
                              {user.email}
                            </Typography.Text>
                          )}
                        </div>
                      </Space>
                      <Divider style={{ marginBlock: 4 }} />
                    </>
                  )}
                  <Menu className={styles.menu} items={menuItems} />
                </Card>
              );
            }}
          >
            <Avatar
              style={{ cursor: 'pointer' }}
              icon={<UserOutlined style={{ fontSize: 14 }} />}
            />
          </Dropdown>
        </Space>
      </StyledTopbar>
    </>
  );
};
