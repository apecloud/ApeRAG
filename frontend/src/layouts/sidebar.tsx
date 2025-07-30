import { SIDEBAR_WIDTH, TOPBAR_HEIGHT } from '@/constants';
import { getLogo } from '@/utils';
import { AppstoreOutlined } from '@ant-design/icons';
import { GlobalToken, Image, theme, Tooltip } from 'antd';
import { BsChatText, BsFiletypeDoc, BsGear } from 'react-icons/bs';
import { css, Link, styled, useIntl, useLocation, useModel } from 'umi';
import UrlPattern from 'url-pattern';

const StyledSidebar = styled('aside').withConfig({
  shouldForwardProp: (prop) => !['token', 'topbar'].includes(prop),
})<{
  token: GlobalToken;
  topbar: boolean;
}>`
  ${({ token, topbar }) => {
    return css`
      width: ${SIDEBAR_WIDTH}px;
      overflow: auto;
      border-right: 1px solid ${token.colorBorderSecondary};
      position: fixed;
      text-align: center;
      left: 0px;
      top: ${topbar ? TOPBAR_HEIGHT : 0}px;
      bottom: 0;
      z-index: ${token.zIndexPopupBase};
      display: flex;
      justify-content: space-between;
      flex-direction: column;
    `;
  }}
`;

const StyledSidebarLink = styled(Link).withConfig({
  shouldForwardProp: (prop) => !['token', 'active'].includes(prop),
})<{
  token: GlobalToken;
  active: boolean;
}>`
  ${({ token, active }) => {
    const gap = 6;
    const size = SIDEBAR_WIDTH - gap * 2;
    return css`
      color: ${active ? token.colorPrimary : token.colorTextSecondary};
      background: ${active ? token.controlItemBgActive : 'none'};
      display: flex;
      width: ${size}px;
      height: ${size}px;
      border-radius: ${gap}px;
      margin: ${gap}px;
      transition: all 0.3s;
      padding: 8px 4px;
      transition: all 0.3s;
      flex-direction: column;
      align-items: center;
      gap: 8px;
      &:hover {
        color: ${active ? token.colorPrimary : token.colorText};
        background: ${token.controlItemBgActive};
      }
      > svg {
        font-size: 18px;
        line-height: 18px;
      }
      > div {
        font-size: 11px;
        line-height: 11px;
      }
    `;
  }}
`;

export const Sidebar = ({ topbar }: { topbar: boolean }) => {
  const { token } = theme.useToken();
  const { formatMessage } = useIntl();
  const { themeName } = useModel('global');
  const { user } = useModel('user');
  const location = useLocation();

  const sidebar_items = [
    {
      path: '/agent',
      icon: <BsChatText />,
      label: formatMessage({ id: 'bot.type_agent' }),
    },
    {
      path: '/bots',
      icon: <AppstoreOutlined />,
      label: formatMessage({ id: 'bot.name_short' }),
    },
    {
      path: '/collections',
      icon: <BsFiletypeDoc />,
      label: formatMessage({ id: 'collection.name' }),
    },
  ];

  return (
    <StyledSidebar token={token} topbar={topbar}>
      <div>
        {!topbar && (
          <Link
            to="/"
            style={{ marginTop: 12, marginBottom: 19, display: 'block' }}
          >
            <Image
              style={{ height: 25, width: 30, marginLeft: 2 }}
              src={getLogo(themeName)}
              preview={false}
            />
          </Link>
        )}

        <div>
          {sidebar_items.map((item, index) => {
            const pattern = new UrlPattern(`${item.path}*`);
            const active = pattern.match(location.pathname);
            return (
              <StyledSidebarLink
                key={index}
                token={token}
                active={active}
                to={item.path}
              >
                {item.icon}
                <div>{item.label}</div>
              </StyledSidebarLink>
            );
          })}
        </div>
      </div>
      {user?.role === 'admin' && (
        <Tooltip
          title={formatMessage({ id: 'action.settings' })}
          placement="right"
        >
          <div>
            <StyledSidebarLink
              token={token}
              active={new UrlPattern(`/settings*`).match(location.pathname)}
              to="/settings"
            >
              <BsGear />
              <div>{formatMessage({ id: 'action.settings' })}</div>
            </StyledSidebarLink>
          </div>
        </Tooltip>
      )}
    </StyledSidebar>
  );
};
