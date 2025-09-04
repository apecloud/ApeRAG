'use client';

import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb';
import { cn } from '@/lib/utils';
import Link from 'next/link';
import React, { useMemo } from 'react';
import {
  AppDocs,
  AppGithub,
  AppLocaleDropdownMenu,
  AppThemeDropdownMenu,
} from './app-topbar';
import { Separator } from './ui/separator';
import { SidebarTrigger, useSidebar } from './ui/sidebar';

export type AppTopbarBreadcrumbItem = {
  title: string;
  href?: string;
};

export const PageHeader = ({
  fixed = true,
  breadcrumbs = [],
  extra,
}: {
  fixed?: boolean;
  breadcrumbs?: AppTopbarBreadcrumbItem[];
  extra?: React.ReactNode;
}) => {
  const { open, isMobile } = useSidebar();

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
          {extra !== undefined ? (
            extra
          ) : (
            <>
              <AppGithub />
              <AppDocs />
              <AppLocaleDropdownMenu />
              <AppThemeDropdownMenu />
            </>
          )}
        </div>
      </header>
      {fixed && <div className="h-12" />}
    </>
  );
};

export const PageTitle = ({
  className,
  ...props
}: React.ComponentProps<'h1'>) => {
  return <h1 className={cn('text-2xl font-medium', className)} {...props} />;
};

export const PageDescription = ({
  className,
  ...props
}: React.ComponentProps<'div'>) => {
  return (
    <div className={cn('text-muted-foreground mb-4', className)} {...props} />
  );
};

export const PageContent = ({
  className,
  ...props
}: React.ComponentProps<'div'>) => {
  return <div className={cn('mx-auto max-w-6xl p-4', className)} {...props} />;
};

export const PageContainer = ({
  className,
  ...props
}: React.ComponentProps<'div'>) => {
  return <div className={cn('', className)} {...props} />;
};
