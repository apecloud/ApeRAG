'use client';

import {
  SidebarGroup,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from '@/components/ui/sidebar';
import { BookOpen, LayoutGrid } from 'lucide-react';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

export const MenuMain = () => {
  const pathname = usePathname();
  const sidebar_workspace = useTranslations('sidebar_workspace');
  return (
    <SidebarGroup className="py-0">
      <SidebarGroupLabel>{sidebar_workspace('repositories')}</SidebarGroupLabel>
      <SidebarMenu>
        <SidebarMenuItem>
          <SidebarMenuButton
            className="data-[active=true]:font-normal"
            asChild
            isActive={pathname.match('/workspace/market/collections') !== null}
          >
            <Link href="/workspace/market/collections">
              <LayoutGrid />
              {sidebar_workspace('marketplace')}
            </Link>
          </SidebarMenuButton>
        </SidebarMenuItem>
        <SidebarMenuItem>
          <SidebarMenuButton
            className="data-[active=true]:font-normal"
            asChild
            isActive={pathname.match('/workspace/collections') !== null}
          >
            <Link href="/workspace/collections">
              <BookOpen />
              {sidebar_workspace('collections')}
            </Link>
          </SidebarMenuButton>
        </SidebarMenuItem>
      </SidebarMenu>
    </SidebarGroup>
  );
};
