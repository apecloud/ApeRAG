import { AppLogo } from '@/components/app-topbar';
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarInset,
  SidebarProvider,
} from '@/components/ui/sidebar';
import { listBotChats, listBots } from '@/features/bot/server-api';
import type { Chat } from '@/features/bot/types';
import { getCurrentUser } from '@/features/auth/server-api';
import { toJson } from '@/lib/utils';
import { redirect } from 'next/navigation';

import { SideBarMenuChats } from '@/components/chat/sidebar-menu-chats';
import { BotProvider } from '@/components/providers/bot-provider';
import { MenuFooter } from './menu-footer';
import { MenuMain } from './menu-main';

export default async function Layout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const user = await getCurrentUser();

  if (!user) {
    redirect(`/auth/signin?callbackUrl=${encodeURIComponent('/workspace')}`);
  }

  const botsRes = await listBots();
  const bot = botsRes.items?.find((item) => item.type === 'agent') ?? undefined;
  let chats: Chat[] = [];

  if (bot?.id) {
    const chatsRes = await listBotChats(bot.id, { page: 1, pageSize: 100 });
    chats = chatsRes.items ?? [];
  }

  return (
    <BotProvider
      workspace={true}
      bot={bot ? toJson(bot) : undefined}
      chats={toJson(chats)}
    >
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
    </BotProvider>
  );
}
