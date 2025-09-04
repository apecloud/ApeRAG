import { Chat } from '@/api';
import { AppLogo } from '@/components/app-topbar';
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarInset,
  SidebarProvider,
} from '@/components/ui/sidebar';
import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';
import { notFound, redirect } from 'next/navigation';

import { SideBarMenuChats } from '@/components/chat/sidebar-menu-chats';
import { AgentsProvider } from '@/components/providers/agents-provider';
import { MenuFooter } from './menu-footer';
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
    <AgentsProvider workspace={true} bot={toJson(bot)} chats={toJson(chats)}>
      <SidebarProvider>
        <Sidebar>
          <SidebarHeader className="h-16 flex-row items-center gap-4 px-4 align-middle">
            <AppLogo />
          </SidebarHeader>
          <SidebarContent className="gap-0">
            <MenuMain />
            <SideBarMenuChats />
          </SidebarContent>

          <MenuFooter />
        </Sidebar>
        <SidebarInset>{children}</SidebarInset>
      </SidebarProvider>
    </AgentsProvider>
  );
}
