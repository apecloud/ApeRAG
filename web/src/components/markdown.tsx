'use client';

import { cn } from '@/lib/utils';
import { ImageIcon } from 'lucide-react';
import Link from 'next/link';
import { useParams, usePathname } from 'next/navigation';
import {
  JSX,
  MouseEventHandler,
  useCallback,
  useMemo,
} from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import rehypeHighlightLines from 'rehype-highlight-code-lines';
import rehypeRaw from 'rehype-raw';
import remarkDirective from 'remark-directive';
import remarkFrontmatter from 'remark-frontmatter';
import remarkGfm from 'remark-gfm';
import remarkGithubAdmonitionsToDirectives from 'remark-github-admonitions-to-directives';
import remarkHeaderId from 'remark-heading-id';
import remarkMdxFrontmatter from 'remark-mdx-frontmatter';
import { AnchorLink } from './anchor-link';
import { ChartMermaid } from './chart-mermaid';
import { Skeleton } from './ui/skeleton';
import { Table, TableBody, TableCell, TableHeader, TableRow } from './ui/table';
import { Tooltip, TooltipContent, TooltipTrigger } from './ui/tooltip';

import './markdown.css';

type AssetContext = {
  collectionId?: string;
  documentId?: string;
  mode?: 'workspace' | 'marketplace';
};

const getRouteParam = (value: string | string[] | undefined) => {
  if (Array.isArray(value)) {
    return value[0];
  }
  return value;
};

const parseAssetUrl = (src: string) => {
  const [assetId = '', queryString = ''] = src.replace('asset://', '').split('?');
  const params = new URLSearchParams(queryString);

  return {
    assetId,
    collectionId: params.get('collection_id') || undefined,
    documentId: params.get('document_id') || undefined,
  };
};

const EXTERNAL_URL_PATTERN = /^https?:\/\//i;
const CJK_URL_TERMINATOR_PATTERN = /[）。，；：！？、》】」』（]/u;
const TRAILING_URL_PUNCTUATION_PATTERN = /[\s)\]}.,;:!?，。；：！？、）】》」』]+$/u;

const normalizeMarkdownHref = (href?: string) => {
  let url = href?.replace(/\.md(?=($|[#?]))/, '') || '/';

  try {
    const decoded = decodeURI(url);
    if (EXTERNAL_URL_PATTERN.test(decoded)) {
      const terminatorIndex = decoded.search(CJK_URL_TERMINATOR_PATTERN);
      if (terminatorIndex > 0) {
        url = decoded.slice(0, terminatorIndex);
      } else {
        url = decoded;
      }
    }
  } catch {
    // Keep the original href if it is not a valid percent-encoded string.
  }

  return url.trim().replace(TRAILING_URL_PUNCTUATION_PATTERN, '') || '/';
};

export const buildDocumentAssetUrl = (src: string, context?: AssetContext) => {
  if (!src.startsWith('asset://')) {
    return src;
  }

  const { assetId, collectionId: queryCollectionId, documentId: queryDocumentId } =
    parseAssetUrl(src);

  const collectionId = queryCollectionId || context?.collectionId;
  const documentId = queryDocumentId || context?.documentId;

  if (!assetId || !collectionId || !documentId) {
    return undefined;
  }

  const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';
  const routePrefix =
    context?.mode === 'marketplace' ? 'marketplace/collections' : 'collections';
  const query = new URLSearchParams({
    path: `assets/${assetId}`,
  });

  return `${basePath}/api/v1/${routePrefix}/${collectionId}/documents/${documentId}/object?${query.toString()}`;
};

const securityLink = (props: JSX.IntrinsicElements['a']) => {
  const url = normalizeMarkdownHref(props.href);
  const isExternal = EXTERNAL_URL_PATTERN.test(url);
  const target = isExternal ? '_blank' : '_self';

  const isNavLink = props.className?.includes('toc-link');
  return isNavLink ? (
    <Tooltip>
      <TooltipTrigger asChild>
        <AnchorLink {...props} href={url || '/'} target={target} />
      </TooltipTrigger>
      <TooltipContent side={isNavLink ? 'left' : 'top'}>
        {props.children}
      </TooltipContent>
    </Tooltip>
  ) : isExternal ? (
    <a
      {...props}
      href={url}
      target={target}
      rel="noopener noreferrer"
      className="underline"
    >
      {props.children}
    </a>
  ) : (
    <Link {...props} href={url || '/'} target={target} className="underline">
      {props.children}
    </Link>
  );
};

const unSecurityLink = (props: JSX.IntrinsicElements['a']) => {
  const url = normalizeMarkdownHref(props.href);
  const isExternal = EXTERNAL_URL_PATTERN.test(url);
  const isNavLink = props.className?.includes('toc-link');
  const handleLinkClick: MouseEventHandler<HTMLAnchorElement> = (e) => {
    if (!isExternal) {
      e.preventDefault();
      e.stopPropagation();
    }
  };
  return isNavLink ? (
    <Tooltip>
      <TooltipTrigger asChild>
        <AnchorLink
          {...props}
          href={url}
          target="_blank"
          onClick={handleLinkClick}
        />
      </TooltipTrigger>
      <TooltipContent side={isNavLink ? 'left' : 'top'}>
        {props.children}
      </TooltipContent>
    </Tooltip>
  ) : isExternal ? (
    <a
      {...props}
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      className="underline"
      onClick={handleLinkClick}
    >
      {props.children}
    </a>
  ) : (
    <Link
      {...props}
      href={url}
      target="_blank"
      className="underline"
      onClick={handleLinkClick}
    >
      {props.children}
    </Link>
  );
};

export const CustomImage = ({
  src,
  resolveAssetUrl,
  ...props
}: JSX.IntrinsicElements['img'] & {
  resolveAssetUrl?: (src: string) => string | undefined;
}) => {
  const imageUrl = useMemo(() => {
    if (typeof src !== 'string') {
      return undefined;
    }

    if (src.startsWith('asset://')) {
      return resolveAssetUrl?.(src);
    }

    return src;
  }, [resolveAssetUrl, src]);

  return imageUrl ? (
    <img
      {...props}
      alt={props.alt}
      src={imageUrl}
      loading="lazy"
      style={{ maxWidth: '100%', height: 'auto' }}
    />
  ) : (
    <Skeleton className="my-4 h-[125px] w-full rounded-xl py-4 pt-8 text-center">
      <ImageIcon className="mx-auto size-12 opacity-20" />
    </Skeleton>
  );
};

const createMdComponents = (resolveAssetUrl?: (src: string) => string | undefined) => ({
  h1: (props: JSX.IntrinsicElements['h1']) => (
    <h1 className="my-6 text-5xl font-bold first:mt-0 last:mb-0">
      {props.children}
    </h1>
  ),
  h2: (props: JSX.IntrinsicElements['h2']) => (
    <h2 className="my-5 text-4xl font-bold first:mt-0 last:mb-0">
      {props.children}
    </h2>
  ),
  h3: (props: JSX.IntrinsicElements['h3']) => (
    <h3 className="my-4 text-3xl font-bold first:mt-0 last:mb-0">
      {props.children}
    </h3>
  ),
  h4: (props: JSX.IntrinsicElements['h4']) => (
    <h4 className="my-3 text-2xl font-bold first:mt-0 last:mb-0">
      {props.children}
    </h4>
  ),
  h5: (props: JSX.IntrinsicElements['h5']) => (
    <h5 className="my-2 text-xl font-bold first:mt-0 last:mb-0">
      {props.children}
    </h5>
  ),
  h6: (props: JSX.IntrinsicElements['h6']) => (
    <h6 className="my-2 text-lg font-bold first:mt-0 last:mb-0">
      {props.children}
    </h6>
  ),
  p: (props: JSX.IntrinsicElements['p']) => (
    <div className="my-2 first:mt-0 last:mb-0">{props.children}</div>
  ),
  blockquote: ({
    className,
    ...props
  }: JSX.IntrinsicElements['blockquote']) => {
    return (
      <blockquote
        className={cn(
          'text-muted-foreground my-4 border-l-4 py-1 pl-4 first:mt-0 last:mb-0',
          className,
        )}
      >
        {props.children}
      </blockquote>
    );
  },
  img: ({ src, ...props }: JSX.IntrinsicElements['img']) => {
    if (!src) {
      return (
        <Skeleton className="my-4 h-[125px] w-full rounded-xl py-4 pt-8 text-center">
          <ImageIcon className="mx-auto size-12 opacity-20" />
        </Skeleton>
      );
    } else if (typeof src === 'string' && src.startsWith('asset://')) {
      return (
        <CustomImage
          src={src}
          {...props}
          resolveAssetUrl={resolveAssetUrl}
        />
      );
    } else {
      return (
        <img
          src={src}
          width={props.width}
          height={props.height}
          alt={props.alt}
          title={props.title}
          style={{ maxWidth: '100%', height: 'auto' }}
        />
      );
    }
  },
  pre: ({ className, ...props }: JSX.IntrinsicElements['pre']) => {
    return (
      <pre className={cn('my-4 overflow-x-auto', className)}>
        {props.children}
      </pre>
    );
  },
  code: ({ className, ...props }: JSX.IntrinsicElements['code']) => {
    const match = /language-(\w+)/.exec(className || '');
    const language = match?.[1];
    if (language) {
      if (language === 'mermaid') {
        return (
          <ChartMermaid>
            {typeof props.children === 'string' ? props.children : ''}
          </ChartMermaid>
        );
      } else {
        return (
          <code className={cn('rounded-md text-sm', className)}>
            {props.children}
          </code>
        );
      }
    } else {
      return (
        <code
          className={cn(
            'mx-1 inline-block overflow-x-auto rounded-md bg-gray-500/10 px-1.5 py-0.5 align-middle text-sm',
            className,
          )}
        >
          {props.children}
        </code>
      );
    }
  },
  ol: ({ className, ...props }: JSX.IntrinsicElements['ul']) => {
    return (
      <ul className={cn('my-4 list-decimal pl-4', className)}>
        {props.children}
      </ul>
    );
  },
  ul: ({ className, ...props }: JSX.IntrinsicElements['ul']) => {
    return (
      <ul className={cn('my-4 list-disc pl-4', className)}>{props.children}</ul>
    );
  },
  li: ({ className, ...props }: JSX.IntrinsicElements['li']) => {
    return (
      <li className={cn('my-1 list-item', className)}>{props.children}</li>
    );
  },
  nav: (props: JSX.IntrinsicElements['nav']) => {
    if (props.className === 'toc') {
      return <nav {...props} />;
    } else {
      return <nav {...props} />;
    }
  },
  table: (props: JSX.IntrinsicElements['table']) => (
    <div className="my-4 overflow-hidden rounded-lg border">
      <Table {...props} />
    </div>
  ),
  thead: (props: JSX.IntrinsicElements['thead']) => <TableHeader {...props} />,
  tbody: (props: JSX.IntrinsicElements['tbody']) => <TableBody {...props} />,
  tr: (props: JSX.IntrinsicElements['tr']) => <TableRow {...props} />,
  td: (props: JSX.IntrinsicElements['td']) => (
    <TableCell>{props.children}</TableCell>
  ),
  th: (props: JSX.IntrinsicElements['th']) => (
    <TableCell>{props.children}</TableCell>
  ),
});

// Keep the legacy export for MDX consumers that do not need route-aware asset resolution.
export const mdComponents = createMdComponents();

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const mdRehypePlugins: any = [
  rehypeRaw,
  rehypeHighlight,
  rehypeHighlightLines,
];

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const mdRemarkPlugins: any = [
  remarkGfm,
  remarkFrontmatter,
  remarkMdxFrontmatter,
  remarkGithubAdmonitionsToDirectives,
  remarkDirective,
  [
    remarkHeaderId,
    {
      defaults: true,
    },
  ],
];

export const Markdown = ({
  rehypeToc = false,
  security = false,
  children,
}: {
  rehypeToc?: boolean;
  security?: boolean;
  children?: string;
}) => {
  const pathname = usePathname();
  const params = useParams<{ collectionId?: string; documentId?: string }>();

  const rehypePlugins = useMemo(() => {
    const plugins = [...mdRehypePlugins];
    if (rehypeToc) {
      plugins.push([
        rehypeToc,
        {
          headings: ['h2', 'h3', 'h4', 'h5', 'h6'],
        },
      ]);
    }
    return plugins;
  }, [rehypeToc]);

  const assetContext = useMemo(
    () => ({
      collectionId: getRouteParam(params?.collectionId),
      documentId: getRouteParam(params?.documentId),
      mode: pathname?.startsWith('/marketplace/')
        ? ('marketplace' as const)
        : ('workspace' as const),
    }),
    [params?.collectionId, params?.documentId, pathname],
  );

  const resolveAssetUrl = useCallback(
    (src: string) => buildDocumentAssetUrl(src, assetContext),
    [assetContext],
  );

  const components = useMemo(
    () => ({
      a: security ? securityLink : unSecurityLink,
      ...createMdComponents(resolveAssetUrl),
    }),
    [resolveAssetUrl, security],
  );

  return (
    <ReactMarkdown
      rehypePlugins={rehypePlugins}
      remarkPlugins={mdRemarkPlugins}
      urlTransform={(url) => url}
      components={components}
    >
      {children}
    </ReactMarkdown>
  );
};
