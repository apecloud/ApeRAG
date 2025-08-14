import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb';
import { House, LucideIcon } from 'lucide-react';
import Link from 'next/link';
import React from 'react';
import { AppGithub, AppThemeDropdownMenu } from './app-topbar';
import { Separator } from './ui/separator';
import { SidebarTrigger } from './ui/sidebar';

export type AppTopbarBreadcrumbItem = {
  icon?: LucideIcon;
  title?: string;
  href?: string;
};

export const PageHeader = ({
  title,
  description,
  breadcrumbs = [],
  children
}: {
  title?: string;
  description?: string;
  breadcrumbs?: AppTopbarBreadcrumbItem[];
  children?: React.ReactNode;
}) => {
  return (
    <>
      <header className="flex h-12 items-center gap-2 border-b transition-[width,height] ease-linear">
        <div className="flex w-full items-center gap-1 px-4 lg:gap-2 lg:px-6">
          <SidebarTrigger className="-ml-1" />
          <Separator
            orientation="vertical"
            className="mx-2 data-[orientation=vertical]:h-4"
          />

          <Breadcrumb>
            <BreadcrumbList>
              <BreadcrumbItem>
                <BreadcrumbLink asChild>
                  <Link href="/" className="flex flex-row items-center  gap-1">
                    <House className='size-4' />
                    Home
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
          <AppThemeDropdownMenu />
        </div>
      </header>
      <div className="flex flex-row items-center px-4 pt-4">
        <h1 className="text-2xl font-medium truncate max-w-80">{title}</h1>
        <div className='ml-auto'>
          {children}
        </div>
      </div>
      {description && <div className="text-muted-foreground px-4">{description}</div>}
    </>
  );
};
