'use client';

import { cn } from '@/lib/utils';
import Link from 'next/link';
import { animateScroll } from 'react-scroll';

export function AnchorLink({
  href,
  className,
  ...props
}: React.ComponentProps<'a'> & {
  href: string;
}) {
  return (
    <Link
      {...props}
      className={cn('underline', className)}
      href={href}
      onClick={(e) => {
        const anchorId =
          typeof href === 'string' && href?.match(/^#/)
            ? href.replace(/^#/, '')
            : undefined;
        const anchor = anchorId && document.getElementById(anchorId);
        const top = anchor && anchor.offsetTop;
        if (top) {
          animateScroll.scrollTo(top - 75, {
            duration: 300,
            smooth: true,
          });
          e.preventDefault();
        }
      }}
    />
  );
}
