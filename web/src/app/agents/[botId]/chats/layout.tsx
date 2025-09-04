import { Chat } from '@/api';

import { AppLogo, AppUserDropdownMenu } from '@/components/app-topbar';
import { MenuChats } from '@/components/chat/menu-chats';
import { AgentsProvider } from '@/components/providers/agents-provider';
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarInset,
  SidebarProvider,
  SidebarSeparator,
} from '@/components/ui/sidebar';
import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';
import { Metadata } from 'next';
import { getTranslations } from 'next-intl/server';
import { notFound, redirect } from 'next/navigation';

export async function generateMetadata(): Promise<Metadata> {
  const page_chat = await getTranslations('page_chat');
  return {
    title: page_chat('metadata.title'),
  };
}

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
  const botsRes = await apiServer.defaultApi.botsBotIdGet({
    botId,
  });
  const bot = botsRes.data;
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
    <AgentsProvider workspace={false} bot={toJson(bot)} chats={toJson(chats)}>
      <SidebarProvider>
        <Sidebar>
          <SidebarHeader className="h-16 flex-row items-center gap-4 px-4 align-middle">
            <AppLogo />
          </SidebarHeader>
          <SidebarContent className="gap-0">
            <MenuChats />
          </SidebarContent>
          <SidebarFooter>
            <SidebarSeparator className="mx-0" />
            <AppUserDropdownMenu />
          </SidebarFooter>
        </Sidebar>
        <SidebarInset>{children}</SidebarInset>
      </SidebarProvider>
    </AgentsProvider>
  );
}
