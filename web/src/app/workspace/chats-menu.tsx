'use client';

import { Bot, Chat } from '@/api';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
} from '@/components/ui/sidebar';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { apiClient } from '@/lib/api/client';
import { EllipsisVertical, Plus, SquarePen, Trash } from 'lucide-react';
import Link from 'next/link';
import { useParams, usePathname, useRouter } from 'next/navigation';
import { useCallback, useState } from 'react';

export const ChatsMenu = ({
  bot,
  chats: initChats,
}: {
  bot: Bot;
  chats: Chat[];
}) => {
  const [chats, setChats] = useState<Chat[]>(initChats);
  const pathname = usePathname();
  const params = useParams();
  const router = useRouter();

  const loadChats = useCallback(async () => {
    if (!bot?.id) return;
    const chatsRes = await apiClient.defaultApi.botsBotIdChatsGet({
      botId: bot.id,
    });
    setChats(chatsRes.data.items || []);
  }, [bot?.id]);

  const handleDelete = useCallback(
    async (chat: Chat) => {
      if (!chat.bot_id || !chat.id) return;
      await apiClient.defaultApi.botsBotIdChatsChatIdDelete({
        botId: chat.bot_id,
        chatId: chat.id,
      });

      if (params.chatId === chat.id) {
        const item = chats?.find((c) => c.id !== chat.id);
        if (item) {
          router.push(`/workspace/agents/${item.bot_id}/chats/${item.id}`);
        } else {
          router.push('/workspace/collections');
        }
      }
      loadChats();
    },
    [chats, loadChats, params.chatId, router],
  );

  const handleCreate = useCallback(async () => {
    if (!bot?.id) return;
    const res = await apiClient.defaultApi.botsBotIdChatsPost({
      botId: bot.id,
      chatCreate: {
        title: '',
      },
    });

    if (res.data.id) {
      router.push(`/workspace/agents/${bot.id}/chats/${res.data.id}`);
      loadChats();
    }
  }, [bot?.id, loadChats, router]);

  return (
    <SidebarGroup>
      <SidebarGroupLabel className="mb-1 flex flex-row justify-between pr-0">
        <span>Chats</span>
        {chats.length > 0 && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Button className="-mr-0.5 size-8" onClick={handleCreate}>
                <Plus />
                <span className="sr-only">Create chat</span>
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right">Create chat</TooltipContent>
          </Tooltip>
        )}
      </SidebarGroupLabel>
      <SidebarGroupContent>
        <SidebarMenu>
          {chats.length > 0 ? (
            chats.map((chat) => {
              const url = `/workspace/agents/${bot?.id}/chats/${chat.id}`;
              return (
                <DropdownMenu key={chat.id}>
                  <SidebarMenuItem>
                    <SidebarMenuButton asChild isActive={pathname === url}>
                      <Link href={url}>{chat.title}</Link>
                    </SidebarMenuButton>
                    <DropdownMenuTrigger asChild>
                      <SidebarMenuAction className="data-[state=open]:bg-accent">
                        <EllipsisVertical className="text-muted-foreground" />
                      </SidebarMenuAction>
                    </DropdownMenuTrigger>

                    <DropdownMenuContent side="right" align="start">
                      <DropdownMenuItem>
                        <SquarePen /> Rename
                      </DropdownMenuItem>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem
                        variant="destructive"
                        onClick={() => handleDelete(chat)}
                      >
                        <Trash /> Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </SidebarMenuItem>
                </DropdownMenu>
              );
            })
          ) : (
            <SidebarMenuItem>
              <SidebarMenuButton
                onClick={handleCreate}
                className="bg-primary text-primary-foreground hover:bg-primary/90 hover:text-primary-foreground active:bg-primary/90 active:text-primary-foreground min-w-8 duration-200 ease-linear"
              >
                <Plus />
                <span>Create chat</span>
              </SidebarMenuButton>
            </SidebarMenuItem>
          )}
        </SidebarMenu>
      </SidebarGroupContent>
    </SidebarGroup>
  );
};
