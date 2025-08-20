import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb';
import { LucideIcon } from 'lucide-react';
import Link from 'next/link';
import React from 'react';
import { Separator } from './ui/separator';
import { SidebarTrigger } from './ui/sidebar';

export type AppTopbarBreadcrumbItem = {
  icon?: LucideIcon;
  title?: string;
  href?: string;
};

export const PageHeader = ({
  title,
  breadcrumbs = [],
}: {
  title?: string;
  breadcrumbs?: AppTopbarBreadcrumbItem[];
  children?: React.ReactNode;
}) => {
  return (
    <header className="flex h-12 shrink-0 items-center gap-2 border-b transition-[width,height] ease-linear">
      <div className="flex w-full items-center gap-1 px-4 lg:gap-2 lg:px-6">
        <SidebarTrigger className="-ml-1" />
        <Separator
          orientation="vertical"
          className="mx-2 data-[orientation=vertical]:h-4"
        />
        <h1 className="text-base font-medium">{title}</h1>
        {breadcrumbs.length ? (
          <Breadcrumb className="p-4 pt-0">
            <BreadcrumbList>
              {breadcrumbs.map((item, index) => {
                const isLast = index === breadcrumbs.length - 1;
                return (
                  <React.Fragment key={index}>
                    <BreadcrumbItem className="flex flex-row items-center gap-1">
                      <BreadcrumbLink asChild>
                        <Link href={item.href || '#'}>{item.title}</Link>
                      </BreadcrumbLink>
                    </BreadcrumbItem>
                    {!isLast && <BreadcrumbSeparator />}
                  </React.Fragment>
                );
              })}
            </BreadcrumbList>
          </Breadcrumb>
        ) : null}
      </div>
    </header>
  );
};
