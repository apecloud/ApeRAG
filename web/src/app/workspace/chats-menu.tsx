'use client';

import { Button } from '@/components/ui/button';
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
import { useWorkspaceContext } from '@/hooks/use-workspace-context';
import _ from 'lodash';
import { Plus, Trash } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

export const ChatsMenu = () => {
  const { bot, chats, chatCreate, chatDelete } = useWorkspaceContext();
  const pathname = usePathname();
  return (
    <SidebarGroup>
      <SidebarGroupLabel className="mb-1 flex flex-row justify-between pr-0">
        <span>Chats</span>
        {_.size(chats) > 0 && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                className="-mr-0.5 size-8 cursor-pointer"
                onClick={chatCreate}
              >
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
          {_.isEmpty(chats) ? (
            <SidebarMenuItem>
              <SidebarMenuButton
                onClick={chatCreate}
                className="bg-primary text-primary-foreground hover:bg-primary/90 hover:text-primary-foreground active:bg-primary/90 active:text-primary-foreground min-w-8 cursor-pointer duration-200 ease-linear"
              >
                <Plus />
                <span>Create chat</span>
              </SidebarMenuButton>
            </SidebarMenuItem>
          ) : (
            chats?.map((chat) => {
              const url = `/workspace/agents/${bot?.id}/chats/${chat.id}`;
              return (
                <SidebarMenuItem key={chat.id} className="group/item">
                  <SidebarMenuButton asChild isActive={pathname === url}>
                    <Link href={url}>
                      <span className="block truncate">{chat.title}</span>
                    </Link>
                  </SidebarMenuButton>
                  <SidebarMenuAction
                    className="invisible cursor-pointer group-hover/item:visible"
                    onClick={() => chatDelete && chatDelete(chat)}
                  >
                    <Trash className="opacity-40 hover:opacity-100" />
                  </SidebarMenuAction>
                </SidebarMenuItem>
              );
            })
          )}
        </SidebarMenu>
      </SidebarGroupContent>
    </SidebarGroup>
  );
};
