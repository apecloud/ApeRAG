import { Navbar, NavbarBody, NavbarHeader } from '@/components';
import { Menu, MenuProps } from 'antd';
import { useMemo } from 'react';
import { FormattedMessage, history, useIntl, useLocation } from 'umi';

type MenuItem = Required<MenuProps>['items'][number];

export const NavbarSettings = () => {
  const location = useLocation();
  const { formatMessage } = useIntl();
  const menuItems = useMemo(
    (): MenuItem[] => [
      {
        label: <FormattedMessage id="users.management" />,
        key: `/settings/users`,
      },
      {
        label: <FormattedMessage id="model.configuration" />,
        key: `/settings/modelConfiguration`,
      },
      {
        label: <FormattedMessage id="apiKeys.title" />,
        key: `/settings/apiKeys`,
      },
      {
        label: <FormattedMessage id="audit.logs.title" />,
        key: `/settings/auditLogs`,
      },
      {
        label: <FormattedMessage id="settings.miscellaneous" />,
        key: `/settings/miscellaneous`,
      },
    ],
    [],
  );

  return (
    <>
      <Navbar>
        <NavbarHeader title={formatMessage({ id: 'system.management' })} />
        <NavbarBody>
          <Menu
            onClick={({ key }) => history.push(key)}
            mode="inline"
            selectedKeys={[location.pathname]}
            items={menuItems}
            style={{
              padding: 0,
              background: 'none',
              border: 'none',
            }}
          />
        </NavbarBody>
      </Navbar>
    </>
  );
};
