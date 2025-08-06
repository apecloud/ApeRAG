import { CSS_PREFIX, THEME_TOKENS } from '@/constants';
import { ReactFlowProvider } from '@xyflow/react';
import { useDebounce, useFavicon, useTitle } from 'ahooks';
import { App, ConfigProvider, theme } from 'antd';
import moment from 'moment';
import { useCallback, useEffect, useMemo, useRef } from 'react';
import { Slide, ToastContainer } from 'react-toastify';
import LoadingBar, { LoadingBarRef } from 'react-top-loading-bar';
import { getLocale, Outlet, useLocation, useModel } from 'umi';
import UrlPattern from 'url-pattern';
import { AuthProvider } from './auth';

import Layout from './layout';

import { LayoutAgent } from '@/pages/agent/LayoutAgent';
import { NavbarAgent } from '@/pages/agent/NavbarAgent';
import { LayoutBot } from '@/pages/bots/$botId/_layout';
import { NavbarBot } from '@/pages/bots/$botId/_navbar';
import { LayoutCollection } from '@/pages/collections/$collectionId/_layout';
import { NavbarCollection } from '@/pages/collections/$collectionId/_navbar';
import { NavbarSettings } from '@/pages/settings/_navbar';

const { darkAlgorithm, defaultAlgorithm } = theme;

/**
 * url pattern rules
 * https://www.npmjs.com/package/url-pattern
 */
const config: {
  [key in string]: React.ReactNode;
} = {
  '/agent(/*)': <Layout navbar={<NavbarAgent />} outlet={<LayoutAgent />} />,

  '/bots/:botId/*': <Layout navbar={<NavbarBot />} outlet={<LayoutBot />} />,
  '/bots(/*)': <Layout />,

  '/collections/:collectionId/*': (
    <Layout navbar={<NavbarCollection />} outlet={<LayoutCollection />} />
  ),
  '/collections(/*)': <Layout />,

  '/marketplace(/*)': <Layout />,

  '/settings(/*)': <Layout navbar={<NavbarSettings />} />,

  '*': <Layout auth={false} sidebar={false} />,
};

export default () => {
  const { themeName, loading } = useModel('global');
  const location = useLocation();
  const ref = useRef<LoadingBarRef>(null);
  const { token } = theme.useToken();
  const algorithm = useMemo(
    () => (themeName.includes('dark') ? darkAlgorithm : defaultAlgorithm),
    [themeName],
  );
  const debouncedLoading = useDebounce(loading, { wait: 500 });

  const getLayout = useCallback(() => {
    const path = Object.keys(config).find((p) =>
      Boolean(new UrlPattern(p).match(location.pathname)),
    );
    return path ? config[path] : <Outlet />;
  }, [location]);

  useEffect(() => {
    if (debouncedLoading) {
      ref.current?.start();
    } else {
      ref.current?.complete();
    }
  }, [debouncedLoading]);

  useTitle(APERAG_CONFIG.title || '');
  useFavicon(APERAG_CONFIG.favicon || '');

  const tokens = THEME_TOKENS[themeName];

  // useEffect(() => {
  //   scroll.scrollToTop({ duration: 100 });
  // }, [location]);

  useEffect(() => {
    const locale = getLocale();
    moment.locale(locale === 'en-US' ? 'en' : 'zh');
  }, []);

  return (
    <AuthProvider>
      <ReactFlowProvider>
        <ConfigProvider
          prefixCls={CSS_PREFIX}
          theme={{
            cssVar: {
              prefix: CSS_PREFIX,
              key: themeName,
            },
            algorithm,
            token: THEME_TOKENS[themeName],
            components: {
              Slider: {
                trackBg: tokens.colorPrimary,
                trackHoverBg: tokens.colorPrimary,
                handleColor: tokens.colorPrimary,
              },
              Menu: {
                itemSelectedColor: tokens.colorText,
              },
            },
          }}
        >
          <App>{getLayout()}</App>
          <LoadingBar ref={ref} color={token.colorPrimary} />
          <ToastContainer
            theme={themeName}
            transition={Slide}
            limit={3}
            autoClose={2000}
            position="bottom-right"
            hideProgressBar
            style={{ fontSize: token.fontSize }}
          />
        </ConfigProvider>
      </ReactFlowProvider>
    </AuthProvider>
  );
};
