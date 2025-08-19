// import rehypeToc from '@jsdevtools/rehype-toc';

import { h } from 'hastscript';
import { JSX } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import rehypeHighlightLines from 'rehype-highlight-code-lines';
import remarkDirective from 'remark-directive';
import remarkFrontmatter from 'remark-frontmatter';
import remarkGfm from 'remark-gfm';
import remarkGithubAdmonitionsToDirectives from 'remark-github-admonitions-to-directives';
import remarkHeaderId from 'remark-heading-id';
import remarkMdxFrontmatter from 'remark-mdx-frontmatter';
import { visit } from 'unist-util-visit';

import { cn } from '@/lib/utils';
import Link from 'next/link';
import { AnchorLink } from './anchor-link';
import { Table, TableBody, TableCell, TableHeader, TableRow } from './ui/table';
import { Tooltip, TooltipContent, TooltipTrigger } from './ui/tooltip';

export const mdComponents = {
  h1: (props: JSX.IntrinsicElements['h1']) => (
    <h1 className="my-6 text-5xl font-bold">{props.children}</h1>
  ),
  h2: (props: JSX.IntrinsicElements['h2']) => (
    <h2 className="my-5 text-4xl font-bold">{props.children}</h2>
  ),
  h3: (props: JSX.IntrinsicElements['h3']) => (
    <h3 className="my-4 text-3xl font-bold">{props.children}</h3>
  ),
  h4: (props: JSX.IntrinsicElements['h4']) => (
    <h4 className="my-3 text-2xl font-bold">{props.children}</h4>
  ),
  h5: (props: JSX.IntrinsicElements['h5']) => (
    <h5 className="my-2 text-xl font-bold">{props.children}</h5>
  ),
  h6: (props: JSX.IntrinsicElements['h6']) => (
    <h6 className="my-2 text-lg font-bold">{props.children}</h6>
  ),
  p: (props: JSX.IntrinsicElements['p']) => (
    <div className="my-1">{props.children}</div>
  ),
  a: (props: JSX.IntrinsicElements['a']) => {
    const target = props.href?.match(/^http/) ? '_blank' : '_self';
    const url = props.href?.replace(/\.md/, '');

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
    ) : (
      <Link
        {...props}
        href={url || '/'}
        target={target}
        className="underline"
      />
    );
  },
  blockquote: ({
    className,
    ...props
  }: JSX.IntrinsicElements['blockquote']) => {
    return (
      <blockquote
        className={cn(
          'text-muted-foreground my-4 border-l-4 py-1 pl-4',
          className,
        )}
        {...props}
      />
    );
  },
  pre: ({ className, children }: JSX.IntrinsicElements['pre']) => {
    return (
      <pre className={cn('my-4 overflow-x-auto', className)}>{children}</pre>
    );
  },
  code: ({ className, ...props }: JSX.IntrinsicElements['code']) => {
    const match = /language-(\w+)/.exec(className || '');
    const language = match?.[1];
    if (language) {
      return (
        <code className={cn('rounded-md text-sm', className)} {...props} />
      );
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
  ul: ({ className, ...props }: JSX.IntrinsicElements['ul']) => {
    return <ul className={cn('my-2 list-disc pl-4', className)} {...props} />;
  },
  li: ({ className, ...props }: JSX.IntrinsicElements['li']) => {
    return <li className={cn('list-item', className)} {...props} />;
  },
  nav: (props: JSX.IntrinsicElements['nav']) => {
    if (props.className === 'toc') {
      return <nav {...props} />;
    } else {
      return <nav {...props} />;
    }
  },
  table: (props: JSX.IntrinsicElements['table']) => (
    <div className="overflow-hidden rounded-lg border">
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
};
export const mdRehypePlugins: any = [
  rehypeHighlight,
  rehypeHighlightLines,
  // [
  //   rehypeToc,
  //   {
  //     headings: ['h2', 'h3', 'h4', 'h5', 'h6'],
  //   },
  // ],
];
export const mdRemarkPlugins: any = [
  remarkGfm,
  remarkFrontmatter,
  remarkMdxFrontmatter,
  remarkGithubAdmonitionsToDirectives,
  remarkDirective,
  () => {
    return (tree: any) => {
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
];

export const Markdown = ({ children }: { children?: string }) => {
  return (
    <ReactMarkdown
      rehypePlugins={mdRehypePlugins}
      remarkPlugins={mdRemarkPlugins}
      components={mdComponents}
    >
      {children}
    </ReactMarkdown>
  );
};
