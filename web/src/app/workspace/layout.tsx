import { Chat } from '@/api';
import { AppLogo, AppUserDropdownMenu } from '@/components/app-topbar';
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarSeparator,
} from '@/components/ui/sidebar';
import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';
import { BatteryMedium, Key, Logs, Package } from 'lucide-react';
import Link from 'next/link';
import { notFound, redirect } from 'next/navigation';

import { WorkspaceProvider } from '@/components/providers/workspace-provider';
import { MenuChats } from './menu-chats';
import { MenuMain } from './menu-main';

export default async function Layout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  let user;
  const apiServer = await getServerApi();

  try {
    const res = await apiServer.defaultApi.userGet();
    user = res.data;
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
  } catch (err) {}

  if (!user) {
    redirect(`/auth/signin?callbackUrl=${encodeURIComponent('/workspace')}`);
  }

  const botsRes = await apiServer.defaultApi.botsGet();
  const bot = botsRes.data.items?.find((item) => item.type === 'agent');
  let chats: Chat[] = [];

  if (!bot) {
    notFound();
  }

  if (bot?.id) {
    const chatsRes = await apiServer.defaultApi.botsBotIdChatsGet({
      botId: bot.id,
      page: 1,
      pageSize: 100,
    });
    //@ts-expect-error api define has a bug
    chats = chatsRes.data.items || [];
  }

  return (
    <WorkspaceProvider bot={toJson(bot)} chats={toJson(chats)}>
      <SidebarProvider>
        <Sidebar>
          <SidebarHeader className="h-16 flex-row items-center gap-4 px-4 align-middle">
            <AppLogo />
          </SidebarHeader>
          <SidebarContent className="gap-0">
            <MenuMain />
            {bot && <MenuChats />}
          </SidebarContent>
          <SidebarFooter className="gap-0">
            <SidebarGroup>
              <SidebarGroupLabel>Settings</SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  <SidebarMenuItem>
                    <SidebarMenuButton
                      asChild
                      className="data-[active=true]:font-normal"
                    >
                      <Link href="/workspace/providers">
                        <Package /> Models
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                  <SidebarMenuItem>
                    <SidebarMenuButton
                      asChild
                      className="data-[active=true]:font-normal"
                    >
                      <Link href="/workspace/api-keys">
                        <Key /> API Keys
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                  <SidebarMenuItem>
                    <SidebarMenuButton
                      asChild
                      className="data-[active=true]:font-normal"
                    >
                      <Link href="/workspace/audit-logs">
                        <Logs /> Audit Logs
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                  <SidebarMenuItem>
                    <SidebarMenuButton
                      asChild
                      className="data-[active=true]:font-normal"
                    >
                      <Link href="/workspace/quotas">
                        <BatteryMedium /> Quotas
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
            <SidebarSeparator className="mx-0 mb-2" />
            <AppUserDropdownMenu />
          </SidebarFooter>
        </Sidebar>
        <SidebarInset>{children}</SidebarInset>
      </SidebarProvider>
    </WorkspaceProvider>
  );
}
