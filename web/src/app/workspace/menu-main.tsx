'use client';

import {
  SidebarGroup,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from '@/components/ui/sidebar';
import { BookOpen, LayoutGrid } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

export const MenuMain = () => {
  const pathname = usePathname();

  return (
    <SidebarGroup className="py-0">
      <SidebarGroupLabel>Repositories</SidebarGroupLabel>
      <SidebarMenu>
        <SidebarMenuItem>
          <SidebarMenuButton
            className="data-[active=true]:font-normal"
            asChild
            isActive={pathname.match('/workspace/market/collections') !== null}
          >
            <Link href="/workspace/market/collections">
              <LayoutGrid />
              Marketplace
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
              Collections
            </Link>
          </SidebarMenuButton>
        </SidebarMenuItem>
      </SidebarMenu>
    </SidebarGroup>
  );
};
