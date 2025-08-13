import type { NextConfig } from 'next';
import createNextIntlPlugin from 'next-intl/plugin';

const nextConfig: NextConfig = {
  /* config options here */

  output: 'standalone',
  poweredByHeader: false,
  experimental: {
    serverActions: {
      bodySizeLimit: '5mb',
    },
  },

  rewrites: async () => {
    return [
      {
        source: '/api/v1/:path*',
        destination: `${process.env.API_ENDPOINT}/api/v1/:path*`,
      },
    ];
  },
};

const withNextIntl = createNextIntlPlugin({
  experimental: {
    // Provide the path to the messages that you're using in `AppConfig`
    createMessagesDeclaration: './src/i18n/en-US.json',
  },
});

export default withNextIntl(nextConfig);
