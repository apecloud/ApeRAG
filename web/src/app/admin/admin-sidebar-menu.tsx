'use client';

import {
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from '@/components/ui/sidebar';
import { BatteryMedium, Logs, MonitorCog, Package } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

export const AdminSideBarMenu = () => {
  const pathname = usePathname();
  return (
    <SidebarGroup>
      <SidebarGroupLabel>Administrator</SidebarGroupLabel>
      <SidebarGroupContent>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              asChild
              isActive={pathname.match('/admin/users') !== null}
            >
              <Link href="/admin/users">
                <Package /> Users
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>

          <SidebarMenuItem>
            <SidebarMenuButton
              asChild
              isActive={pathname.match('/admin/providers') !== null}
            >
              <Link href="/admin/providers">
                <BatteryMedium /> Models
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>

          <SidebarMenuItem>
            <SidebarMenuButton
              asChild
              isActive={pathname.match('/admin/audit-logs') !== null}
            >
              <Link href="/admin/audit-logs">
                <Logs /> Audit Logs
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>

          <SidebarMenuItem>
            <SidebarMenuButton
              asChild
              isActive={pathname.match('/admin/configuration') !== null}
            >
              <Link href="/admin/configuration">
                <MonitorCog /> Configuration
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarGroupContent>
    </SidebarGroup>
  );
};
