'use client';

import { AppTopbarBreadcrumbItem } from '@/components/page-container';
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb';
import { Separator } from '@/components/ui/separator';
import { SidebarTrigger, useSidebar } from '@/components/ui/sidebar';
import { cn } from '@/lib/utils';
import Link from 'next/link';
import React, { useMemo } from 'react';

export const BotHeader = ({
  fixed = true,
  breadcrumbs = [],
  extra,
}: {
  fixed?: boolean;
  breadcrumbs?: AppTopbarBreadcrumbItem[];
  extra?: React.ReactNode;
}) => {
  const { isMobile, open } = useSidebar();

  const cls = useMemo(() => {
    const defaultCls = cn(
      'flex flex-row justify-between h-12 items-center gap-2 border-b transition-[width,height,left] ease-linear bg-background/50 backdrop-blur-lg',
    );

    return fixed
      ? cn(
          defaultCls,
          'fixed right-0 top-0 z-10',
          !open || isMobile ? 'left-0' : 'left-[var(--sidebar-width)]',
        )
      : defaultCls;
  }, [fixed, open, isMobile]);

  return (
    <>
      <header className={cls}>
        <div className="flex w-full items-center gap-1 px-4 lg:gap-2 lg:px-6">
          <SidebarTrigger className="-ml-1 cursor-pointer" />
          <Separator
            orientation="vertical"
            className="mx-2 data-[orientation=vertical]:h-4"
          />
          <Breadcrumb>
            <BreadcrumbList>
              {breadcrumbs.map((item, index) => {
                const isLast = index === breadcrumbs.length - 1;
                return (
                  <React.Fragment key={index}>
                    <BreadcrumbItem className="flex flex-row items-center gap-1">
                      {item.href ? (
                        <BreadcrumbLink asChild className="text-foreground">
                          <Link href={item.href || '#'}>{item.title}</Link>
                        </BreadcrumbLink>
                      ) : (
                        <div>{item.title}</div>
                      )}
                    </BreadcrumbItem>
                    {!isLast && <BreadcrumbSeparator />}
                  </React.Fragment>
                );
              })}
            </BreadcrumbList>
          </Breadcrumb>
        </div>
        <div className="flex flex-row items-center gap-2 pr-4">
          {extra !== undefined ? extra : <></>}
        </div>
      </header>
      {fixed && <div className="h-12" />}
    </>
  );
};
