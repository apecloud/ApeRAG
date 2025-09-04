import { Chat } from '@/api';

import { SideBarMenuChats } from '@/components/chat/sidebar-menu-chats';
import { BotProvider } from '@/components/providers/bot-provider';
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarInset,
  SidebarProvider,
} from '@/components/ui/sidebar';
import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';
import { Bot } from 'lucide-react';
import { notFound, redirect } from 'next/navigation';

export default async function ChatLayout({
  children,
  params,
}: Readonly<{
  children: React.ReactNode;
  params: Promise<{ botId: string }>;
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

  const { botId } = await params;
  const botRes = await apiServer.defaultApi.botsBotIdGet({
    botId,
  });
  let bot = botRes.data;
  let chats: Chat[] = [];

  if (!bot) {
    const createRes = await apiServer.defaultApi.botsPost({
      botCreate: {
        title: 'Default Agent Bot',
        type: 'agent',
      },
    });
    bot = createRes.data;
    if (!botRes.data.id) {
      notFound();
    }
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
    <BotProvider
      mention={false}
      workspace={false}
      bot={toJson(bot)}
      chats={toJson(chats)}
    >
      <SidebarProvider>
        <Sidebar>
          <SidebarHeader className="flex h-16 flex-row items-center gap-2 px-2 align-middle">
            <Bot />
            <div className="flex-1 truncate text-sm font-bold">{bot.title}</div>
          </SidebarHeader>
          <SidebarContent className="gap-0">
            <SideBarMenuChats />
          </SidebarContent>
        </Sidebar>
        <SidebarInset>{children}</SidebarInset>
      </SidebarProvider>
    </BotProvider>
  );
}
