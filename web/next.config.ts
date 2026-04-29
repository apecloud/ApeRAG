import createMDXPlugin from '@next/mdx';
import type { NextConfig } from 'next';
import createNextIntlPlugin from 'next-intl/plugin';
import path from 'node:path';

const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';

const nextConfig: NextConfig = {
  /* config options here */

  // To disable this UI completely, set devIndicators: false in your next.config file.
  // devIndicators: false,

  basePath,

  reactStrictMode: false,

  // Configure `pageExtensions` to include markdown and MDX files
  pageExtensions: ['js', 'jsx', 'md', 'mdx', 'ts', 'tsx'],

  output: 'standalone',
  outputFileTracingRoot: path.join(process.cwd(), '..'),
  poweredByHeader: false,
  experimental: {
    serverActions: {
      bodySizeLimit: '100mb',
    },
  },

  modularizeImports: {},

  transpilePackages: [],

  webpack: (config) => {
    config.resolve ??= {};
    config.resolve.alias = {
      ...(config.resolve.alias ?? {}),
      '@/cosmograph/style.module.css$': path.resolve(
        process.cwd(),
        'node_modules/@cosmograph/cosmograph/cosmograph/style.module.css.js',
      ),
    };

    return config;
  },
};

const withNextIntl = createNextIntlPlugin({
  experimental: {
    // Provide the path to the messages that you're using in `AppConfig`
    createMessagesDeclaration: './src/i18n/en-US.json',
  },
});

/**
 * https://nextjs.org/docs/app/guides/mdx
 * https://github.com/vercel/next.js/issues/71819#issuecomment-2461802968
 *
 * No JS remark/rehype plugins are wired here on purpose: there are
 * currently zero `.mdx` pages in the tree, so any plugin chain would be
 * dead code at compile time. Runtime markdown rendering goes through
 * `web/src/components/markdown.tsx`, which keeps its own React-side
 * `react-markdown` plugin pipeline (gfm, highlight, directives,
 * frontmatter, headerId).
 *
 * Keeping the chain empty makes the loader options worker-serializable,
 * which is what `next dev --turbopack` requires (custom inline closures
 * break the Turbopack worker pool with "loader options are not
 * serializable"). If `.mdx` pages are introduced later, prefer
 * `experimental.mdxRs: true` (Rust-based MDX) for Turbopack
 * compatibility, or extract any JS plugins into their own modules so
 * each one can be passed as an importable reference.
 */
const withNextMDX = createMDXPlugin({
  extension: /\.(md|mdx)$/,
});

export default withNextMDX(withNextIntl(nextConfig));
