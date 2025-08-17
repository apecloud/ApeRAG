import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb';
import { cn } from '@/lib/utils';
import { House } from 'lucide-react';
import Link from 'next/link';
import React from 'react';
import { AppDocs, AppGithub, AppThemeDropdownMenu } from './app-topbar';
import { Separator } from './ui/separator';
import { SidebarTrigger } from './ui/sidebar';

export type AppTopbarBreadcrumbItem = {
  title: string;
  href?: string;
};

export const PageHeader = ({
  breadcrumbs = [],
}: {
  breadcrumbs?: AppTopbarBreadcrumbItem[];
}) => {
  return (
    <header className="flex h-16 items-center gap-2 border-b transition-[width,height] ease-linear">
      <div className="flex w-full items-center gap-1 px-4 lg:gap-2 lg:px-6">
        <SidebarTrigger className="-ml-1 cursor-pointer" />
        <Separator
          orientation="vertical"
          className="mx-2 data-[orientation=vertical]:h-4"
        />
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink asChild>
                <Link
                  href="/workspace"
                  className="text-foreground flex flex-row items-center gap-1"
                >
                  <House className="size-4" />
                </Link>
              </BreadcrumbLink>
            </BreadcrumbItem>

            {breadcrumbs.length > 0 && <BreadcrumbSeparator />}
            {breadcrumbs.map((item, index) => {
              const isLast = index === breadcrumbs.length - 1;
              return (
                <React.Fragment key={index}>
                  <BreadcrumbItem className="flex flex-row items-center gap-1">
                    {item.href ? (
                      <BreadcrumbLink asChild>
                        <Link href={item.href || '#'}>{item.title}</Link>
                      </BreadcrumbLink>
                    ) : (
                      <div className="text-primary">{item.title}</div>
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
        <AppGithub />
        <AppDocs />
        <AppThemeDropdownMenu />
      </div>
    </header>
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
  return (
    <div>
      <div className={cn('mx-auto max-w-7xl p-4', className)} {...props} />
    </div>
  );
};

export const PageContainer = ({
  className,
  ...props
}: React.ComponentProps<'div'>) => {
  return <div className={cn('', className)} {...props} />;
};
