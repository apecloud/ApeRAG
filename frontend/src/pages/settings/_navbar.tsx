import { Navbar, NavbarBody, NavbarHeader } from '@/components';
import { Menu, MenuProps } from 'antd';
import { useMemo } from 'react';
import { FormattedMessage, history, useIntl, useLocation, useModel } from 'umi';

type MenuItem = Required<MenuProps>['items'][number];

export const NavbarSettings = () => {
  const location = useLocation();
  const { formatMessage } = useIntl();
  const { user } = useModel('user');
  
  const menuItems = useMemo(
    (): MenuItem[] => {
      const items: MenuItem[] = [
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
      ];

      // Only show user management and miscellaneous for admin users
      if (user?.role === 'admin') {
        items.unshift({
          label: <FormattedMessage id="users.management" />,
          key: `/settings/users`,
        });
        items.push({
          label: <FormattedMessage id="quota.management" />,
          key: `/settings/quotas`,
        });
        items.push({
          label: <FormattedMessage id="settings.miscellaneous" />,
          key: `/settings/miscellaneous`,
        });
      } else {
        // Regular users can view their own quotas
        items.push({
          label: <FormattedMessage id="quota.management" />,
          key: `/settings/quotas`,
        });
      }

      return items;
    },
    [user?.role],
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
