import { Navbar, NavbarBody, NavbarHeader } from '@/components';
import { Menu, MenuProps } from 'antd';
import { useMemo } from 'react';
import {
  FormattedMessage,
  history,
  useLocation,
  useModel,
  useParams,
} from 'umi';

type MenuItem = Required<MenuProps>['items'][number];

export const NavbarCollection = () => {
  const { collectionId } = useParams();
  const location = useLocation();
  const { collection } = useModel('collection');

  const menuItems = useMemo(
    (): MenuItem[] => [
      {
        label: <FormattedMessage id="collection.files" />,
        key: `/collections/${collectionId}/documents`,
      },
      {
        label: <FormattedMessage id="collection.search" />,
        key: `/collections/${collectionId}/search`,
      },
      {
        label: <FormattedMessage id="collection.settings" />,
        key: `/collections/${collectionId}/settings`,
      },
    ],
    [collectionId],
  );

  if (!collection) return;

  return (
    <Navbar>
      <NavbarHeader title={collection?.title} backTo="/collections" />
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
  );
};
