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
import Link from 'next/link';

export const MenuFooter = () => {
  const { isMobile } = useSidebar();

  return (
    <SidebarFooter>
      <SidebarGroup className="p-0">
        <SidebarGroupLabel>More</SidebarGroupLabel>
        <SidebarGroupContent>
          <SidebarMenu>
            <DropdownMenu>
              <DropdownMenuTrigger
                asChild
                className="data-[state=open]:bg-accent h-auto"
              >
                <SidebarMenuButton className="data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground">
                  <FlaskConical />
                  Evaluation
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
                      <History /> Evaluation
                    </Link>
                  </DropdownMenuItem>
                  <DropdownMenuItem asChild>
                    <Link href="/workspace/evaluations/questions">
                      <MailQuestionMark /> Question Sets
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
                  Settings
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
                      <Package /> Models
                    </Link>
                  </DropdownMenuItem>

                  <DropdownMenuItem asChild>
                    <Link href="/workspace/api-keys">
                      <Key /> API Keys
                    </Link>
                  </DropdownMenuItem>

                  <DropdownMenuItem asChild>
                    <Link href="/workspace/audit-logs">
                      <Logs /> Audit Logs
                    </Link>
                  </DropdownMenuItem>

                  <DropdownMenuItem asChild>
                    <Link href="/workspace/quotas">
                      <BatteryMedium /> Quotas
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
