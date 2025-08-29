'use client';

import { AppUserDropdownMenu } from '@/components/app-topbar';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarSeparator,
  useSidebar,
} from '@/components/ui/sidebar';
import {
  BatteryMedium,
  ChevronRight,
  FlaskConical,
  History,
  Key,
  Logs,
  MailQuestionMark,
  Package,
  Settings,
} from 'lucide-react';
import { useTranslations } from 'next-intl';
import Link from 'next/link';

export const MenuFooter = () => {
  const { isMobile } = useSidebar();
  const sidebar_workspace = useTranslations('sidebar_workspace');

  return (
    <SidebarFooter>
      <SidebarGroup className="p-0">
        <SidebarGroupLabel>{sidebar_workspace('more')}</SidebarGroupLabel>
        <SidebarGroupContent>
          <SidebarMenu>
            <DropdownMenu>
              <DropdownMenuTrigger
                asChild
                className="data-[state=open]:bg-accent h-auto"
              >
                <SidebarMenuButton className="data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground">
                  <FlaskConical />
                  {sidebar_workspace('question_evaluation')}
                  <ChevronRight className="ml-auto" />
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                className="w-(--radix-dropdown-menu-trigger-width) min-w-56 rounded-lg"
                align="end"
                side={isMobile ? 'bottom' : 'right'}
                sideOffset={isMobile ? 4 : 12}
              >
                <DropdownMenuGroup>
                  <DropdownMenuItem asChild>
                    <Link href="/workspace/evaluations">
                      <History /> {sidebar_workspace('evaluations')}
                    </Link>
                  </DropdownMenuItem>
                  <DropdownMenuItem asChild>
                    <Link href="/workspace/evaluations/questions">
                      <MailQuestionMark /> {sidebar_workspace('question_sets')}
                    </Link>
                  </DropdownMenuItem>
                </DropdownMenuGroup>
              </DropdownMenuContent>
            </DropdownMenu>

            <DropdownMenu>
              <DropdownMenuTrigger
                asChild
                className="data-[state=open]:bg-accent h-auto"
              >
                <SidebarMenuButton className="data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground">
                  <Settings />
                  {sidebar_workspace('settings')}
                  <ChevronRight className="ml-auto" />
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                className="w-(--radix-dropdown-menu-trigger-width) min-w-56 rounded-lg"
                align="end"
                side={isMobile ? 'bottom' : 'right'}
                sideOffset={isMobile ? 4 : 12}
              >
                <DropdownMenuGroup>
                  <DropdownMenuItem asChild>
                    <Link href="/workspace/providers">
                      <Package /> {sidebar_workspace('models')}
                    </Link>
                  </DropdownMenuItem>

                  <DropdownMenuItem asChild>
                    <Link href="/workspace/api-keys">
                      <Key /> {sidebar_workspace('api_keys')}
                    </Link>
                  </DropdownMenuItem>

                  <DropdownMenuItem asChild>
                    <Link href="/workspace/audit-logs">
                      <Logs /> {sidebar_workspace('audit_logs')}
                    </Link>
                  </DropdownMenuItem>

                  <DropdownMenuItem asChild>
                    <Link href="/workspace/quotas">
                      <BatteryMedium /> {sidebar_workspace('quotas')}
                    </Link>
                  </DropdownMenuItem>
                </DropdownMenuGroup>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenu>
        </SidebarGroupContent>
      </SidebarGroup>

      <SidebarSeparator className="mx-0" />
      <AppUserDropdownMenu />
    </SidebarFooter>
  );
};
