import path from 'path';
import { defineConfig } from 'umi';

const BASE_PATH = path.join(
  process.env.BASE_PATH ? process.env.BASE_PATH : '',
  '/',
);
const PUBLIC_PATH = BASE_PATH;

export default defineConfig({
  base: BASE_PATH,
  publicPath: PUBLIC_PATH,
  manifest: {
    fileName: 'manifest.json',
    basePath: '',
  },
  conventionRoutes: {
    base: 'src/pages',
    exclude: [/\/models\//, /\/model\/.ts/, /\/_.*/],
  },
  title: 'ApeRAG',
  favicons: [`${PUBLIC_PATH}favicon.ico`],
  metas: [
    {
      name: 'viewport',
      content:
        'width=device-width, initial-scale=1.0, minimum-scale=1.0, maximum-scale=1.0, user-scalable=no',
    },
    { name: 'apple-mobile-web-app-capable', content: 'yes' },
    { name: 'mobile-web-app-capable', content: 'yes' },
  ],
  model: {},
  clientLoader: {},
  define: {
    BASE_PATH,
    PUBLIC_PATH,
  },
  locale: {
    antd: true,
    default: 'en-US',
    baseNavigator: false,
    baseSeparator: '-',
    title: true,
    useLocalStorage: true,
  },

  antd: {},
  headScripts: [`${PUBLIC_PATH}settings.js?t=${new Date().getTime()}`],
  devtool: process.env.NODE_ENV === 'development' ? 'eval' : false,
  esbuildMinifyIIFE: true,
  proxy: {
    '/api': {
      target: 'http://127.0.0.1:8000',
      changeOrigin: true,
      ws: true,
      secure: false,
    },
  },
  cssLoaderModules: {
    exportLocalsConvention: 'camelCase',
  },
  styledComponents: {},
  cssLoader: {},
  jsMinifier: 'terser',
  cssMinifier: 'esbuild',
  cssMinifierOptions: {},
  alias: {},
  codeSplitting: {
    jsStrategy: 'depPerChunk',
    jsStrategyOptions: {},
  },
  hash: true,
  mfsu: false,
  chainWebpack: (config: any) => {
    config.output.chunkFilename('[contenthash:16].js').end();
    config
      .plugin('mini-css-extract-plugin')
      .tap(() => [
        {
          filename: 'umi.[contenthash:8].css',
          chunkFilename: '[contenthash:16].css',
          ignoreOrder: true,
        },
      ])
      .end();
  },
  npmClient: 'yarn',
  plugins: [
    '@umijs/plugins/dist/antd',
    '@umijs/plugins/dist/model',
    '@umijs/plugins/dist/access',
    '@umijs/plugins/dist/locale',
    '@umijs/plugins/dist/styled-components',
  ],
});
