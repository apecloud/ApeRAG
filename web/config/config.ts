import { defineConfig } from '@umijs/max';

const BASE_PATH = process.env.BASE_PATH || '';
const PUBLIC_PATH = process.env.PUBLIC_PATH || '/';
const SITE_LOGO =
  process.env.SITE_LOGO ||
  'https://cdn.kubeblocks.io/img/apecloud/favicon.png';
const SITE_TITLE = process.env.SITE_TITLE || 'ApeRAG';

export default defineConfig({
  antd: {},
  access: {},
  model: {},
  initialState: {},
  base: BASE_PATH,
  publicPath: PUBLIC_PATH,
  outputPath: './build',
  request: {},
  layout: false,
  locale: {
    default: 'en-US',
    antd: true,
    title: true,
    baseNavigator: false,
    baseSeparator: '-',
  },
  favicons: [SITE_LOGO],
  metas: [
    { name: 'viewport', content: 'width=device-width, initial-scale=1.0, minimum-scale=1.0, maximum-scale=1.0, user-scalable=no' },
    { name: 'apple-mobile-web-app-capable', content: 'yes'},
    { name: 'mobile-web-app-capable', content: 'yes'}
  ],
  cssLoader: {},
  cssLoaderModules: {
    exportLocalsConvention: 'camelCase',
    localIdentName: '[path][name]__[local]--[hash:base64:5]',
  },
  jsMinifier: 'terser',
  cssMinifier: 'esbuild',
  cssMinifierOptions: {},
  alias: {},
  codeSplitting: {
    jsStrategy: 'depPerChunk',
    jsStrategyOptions: {},
  },
  headScripts: [
    `
      window.API_ENDPOINT = ${process.env.API_ENDPOINT ? ('"' + process.env.API_ENDPOINT + '"') : undefined};
    `
  ],
  define: {
    NODE_ENV: process.env.NODE_ENV,
    BASE_PATH,
    PUBLIC_PATH,
    SITE_LOGO,
    SITE_TITLE,
  },
  mock: {},
  proxy: {
    '/api': {
      target: process.env.API_ENDPOINT,
      changeOrigin: true,
      ws: true,
    },
  },
  routes: [
    {
      path: '/callback',
      component: './Callback',
    },
    {
      path: '/',
      component: '@/runtimes/customLayout.tsx',
      wrappers: ['@/components/Auth'],
      routes: [
        {
          path: '/settings',
          component: './User/settings',
        },
        {
          path: '/bots',
          routes: [
            {
              path: '/bots/',
              component: './Bots/list',
            },
            {
              path: '/bots/new',
              component: './Bots/new',
            },
            {
              path: '/bots/:botId/chat',
              component: './Chat/go',
            },
            {
              path: '/bots/:botId',
              component: './Bots/detail',
              routes: [
                {
                  path: '/bots/:botId/settings',
                  component: './Bots/settings',
                },
                {
                  path: '/bots/:botId/queries',
                  component: './Bots/queries',
                  routes: [
                    {
                      path: '/bots/:botId/queries/welcome',
                      component: './Bots/queries/welcome',
                    },
                    {
                      path: '/bots/:botId/queries/platform',
                      component: './Bots/queries/platform',
                    },
                  ],
                },
                {
                  path: '/bots/:botId/integrations',
                  component: './Bots/integrations',
                },
              ],
            },
          ],
        },
        {
          path: '/collections',
          routes: [
            {
              path: '/collections/:page',
              component: './Collections/list',
            },
            {
              path: '/collections',
              component: './Collections/list',
            },
            {
              path: '/collections/new',
              component: './Collections/new',
            },
            {
              path: '/collections/:collectionId',
              component: './Collections/detail',
              routes: [
                {
                  path: '/collections/:collectionId/documents/:page',
                  component: './Collections/documents',
                },
                {
                  path: '/collections/:collectionId/documents',
                  component: './Collections/documents',
                },
                {
                  path: '/collections/:collectionId/sync',
                  component: './Collections/sync',
                },
                {
                  path: '/collections/:collectionId/settings',
                  component: './Collections/settings',
                },
                {
                  path: '/collections/:collectionId/questions',
                  component: './Collections/questions',
                },
              ],
            },
          ],
        },
        {
          path: '/modelServiceProviders',
          routes: [
            {
              path: '/modelServiceProviders',
              component: './ModelServiceProviders/list',
            },
            {
              path: '/modelServiceProviders/:modelServiceProviderName/settings',
              component: './ModelServiceProviders/settings',
            }
          ],
        },
      ],
    },
  ],
  npmClient: 'yarn',
  manifest: {
    fileName: 'manifest.json',
    basePath: '',
  },
  esbuildMinifyIIFE: true,
  chainWebpack: (config) => {
    config.output.filename(`[name].[contenthash:8].js`).end();
    config.output.chunkFilename('[contenthash:16].js').end();

    config
      .plugin('mini-css-extract-plugin')
      .tap(() => [
        {
          chunkFilename: '[contenthash:16].css',
          ignoreOrder: true,
        },
      ])
      .end();
  },
});
