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
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

export const AdminSideBarMenu = () => {
  const pathname = usePathname();
  const page_auth = useTranslations('page_auth');
  const admin_users = useTranslations('admin_users');
  const admin_config = useTranslations('admin_config');
  const page_models = useTranslations('page_models');
  const page_audit_logs = useTranslations('page_audit_logs');

  return (
    <SidebarGroup>
      <SidebarGroupLabel>{page_auth('administrator')}</SidebarGroupLabel>
      <SidebarGroupContent>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              asChild
              isActive={pathname.match('/admin/users') !== null}
            >
              <Link href="/admin/users">
                <Package /> {admin_users('metadata.title')}
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>

          <SidebarMenuItem>
            <SidebarMenuButton
              asChild
              isActive={pathname.match('/admin/providers') !== null}
            >
              <Link href="/admin/providers">
                <BatteryMedium /> {page_models('metadata.model_title')}
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>

          <SidebarMenuItem>
            <SidebarMenuButton
              asChild
              isActive={pathname.match('/admin/audit-logs') !== null}
            >
              <Link href="/admin/audit-logs">
                <Logs /> {page_audit_logs('metadata.title')}
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>

          <SidebarMenuItem>
            <SidebarMenuButton
              asChild
              isActive={pathname.match('/admin/configuration') !== null}
            >
              <Link href="/admin/configuration">
                <MonitorCog /> {admin_config('metadata.title')}
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarGroupContent>
    </SidebarGroup>
  );
};
