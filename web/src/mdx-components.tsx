import { mdComponents } from '@/components/markdown';
import type { MDXComponents } from 'mdx/types';

export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    ...mdComponents,
    ...components,
  };
}
