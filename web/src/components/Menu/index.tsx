import { useIntl } from '@umijs/max';
import { useEffect, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';

function Menu({ menuCollapse, setMenuCollapse }) {
  const intl = useIntl();

  const botsText = intl.formatMessage({
    id: 'nav.bots',
  });
  const collectionsText = intl.formatMessage({
    id: 'nav.collections',
  });
  const modelServiceProvidersText = intl.formatMessage({
    id: 'nav.model_service_providers',
  });
  const [activeItem, setActiveItem] = useState('');
  const location = useLocation();

  useEffect(() => {
    const path = location.pathname.substring(1).toLowerCase().split('/');
    setActiveItem(path[0]);
  }, [location]);

  const menuClick = (item) => {
    setActiveItem(item);
    if(window.innerWidth<=820){
      setMenuCollapse(0);
    }
  };

  return (
    <nav id="global-menu" className={menuCollapse?'global-menu':'global-menu global-menu-off'}>
      <ul>
        <li>
          <Link
            className={
              activeItem === '' || activeItem === 'bots' ? 'bots on' : 'bots'
            }
            onClick={() => menuClick('robots')}
            to="/bots"
          >
            { botsText }
          </Link>
        </li>
        <li>
          <Link
            className={
              activeItem === 'collections' ? 'collections on' : 'collections'
            }
            onClick={() => menuClick('collections')}
            to="/collections"
          >
            { collectionsText }
          </Link>
        </li>
        <li>
          <Link
            className={
              activeItem === 'modelServiceProviders' ? 'modelServiceProviders on' : 'modelServiceProviders'
            }
            onClick={() => menuClick('modelServiceProviders')}
            to="/modelServiceProviders"
          >
            { modelServiceProvidersText }
          </Link>
        </li>
      </ul>
    </nav>
  );
}

export default Menu;
