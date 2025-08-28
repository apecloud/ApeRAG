import createMDXPlugin from '@next/mdx';
import { h } from 'hastscript';
import type { NextConfig } from 'next';
import createNextIntlPlugin from 'next-intl/plugin';
import remarkDirective from 'remark-directive';
import remarkFrontmatter from 'remark-frontmatter';
import remarkGfm from 'remark-gfm';
import remarkGithubAdmonitionsToDirectives from 'remark-github-admonitions-to-directives';
import remarkHeaderId from 'remark-heading-id';
import { visit } from 'unist-util-visit';
// import remarkMdxFrontmatter from "remark-mdx-frontmatter";
// import rehypeToc from '@jsdevtools/rehype-toc';
import rehypeHighlight from 'rehype-highlight';

// import rehypeHighlightLines from 'rehype-highlight-code-lines';

const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';
const apiServerEndpoint =
  (process.env.NEXT_PUBLIC_API_SERVER_ENDPOINT || 'http://localhost:8000') +
  (process.env.NEXT_PUBLIC_API_SERVER_BASE_PATH || '/api/v1');

const nextConfig: NextConfig = {
  /* config options here */

  // To disable this UI completely, set devIndicators: false in your next.config file.
  // devIndicators: false,

  basePath,

  reactStrictMode: false,

  // Configure `pageExtensions` to include markdown and MDX files
  pageExtensions: ['js', 'jsx', 'md', 'mdx', 'ts', 'tsx'],

  output: 'standalone',
  poweredByHeader: false,
  experimental: {
    serverActions: {
      bodySizeLimit: '10mb',
    },
  },

  // Will only be available on the server side
  serverRuntimeConfig: {},
  // Will be available on both server and client
  publicRuntimeConfig: {},

  modularizeImports: {},

  transpilePackages: [],

  rewrites: async () => {
    return [
      {
        source: `${process.env.NEXT_PUBLIC_API_SERVER_BASE_PATH}/:path*`,
        destination: `${apiServerEndpoint}/:path*`,
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

/**
 * https://nextjs.org/docs/app/guides/mdx
 * https://github.com/vercel/next.js/issues/71819#issuecomment-2461802968
 */
const withNextMDX = createMDXPlugin({
  // Add markdown plugins here, as desired
  extension: /\.(md|mdx)$/,
  options: {
    remarkPlugins: [
      remarkGfm,
      remarkFrontmatter,
      // remarkMdxFrontmatter,
      remarkGithubAdmonitionsToDirectives,
      remarkDirective,
      () => {
        return (tree) => {
          visit(tree, (node) => {
            if (node.type === 'containerDirective') {
              const data = node.data || (node.data = {});
              const tagName = 'div';
              data.hName = tagName;
              data.hProperties = h(tagName, {
                ...node.attributes,
                class: node.name,
              }).properties;
            }
          });
        };
      },
      [
        remarkHeaderId,
        {
          defaults: true,
        },
      ],
    ],
    rehypePlugins: [
      rehypeHighlight,
      // rehypeHighlightLines,
      // [
      //   rehypeToc,
      //   {
      //     // position: 'beforebegin', // "beforebegin" | "afterbegin" | "beforeend" | "afterend"
      //     headings: ['h2', 'h3', 'h4', 'h5', 'h6'],
      //   },
      // ],
    ],
  },
});

export default withNextMDX(withNextIntl(nextConfig));
