'use client';

import { useAgentsContext } from '@/components/providers/agents-provider';
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
import _ from 'lodash';
import { ArrowLeft, Plus, Trash } from 'lucide-react';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

export const MenuChats = () => {
  const { bot, workspace, chats, chatCreate, chatDelete } = useAgentsContext();
  const pathname = usePathname();
  const sidebar_workspace = useTranslations('sidebar_workspace');
  return (
    <SidebarGroup>
      {workspace ? (
        <SidebarGroupLabel className="mb-1 flex flex-row justify-between pr-0">
          <span>{sidebar_workspace('chats')}</span>
          {_.size(chats) > 0 && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  className="-mr-0.5 size-8 cursor-pointer"
                  onClick={chatCreate}
                  size="icon"
                  variant="secondary"
                >
                  <Plus />
                  <span className="sr-only">
                    {sidebar_workspace('chats_new')}
                  </span>
                </Button>
              </TooltipTrigger>
              <TooltipContent side="right">
                {sidebar_workspace('chats_new')}
              </TooltipContent>
            </Tooltip>
          )}
        </SidebarGroupLabel>
      ) : (
        <div className="mb-2 flex items-center gap-2">
          <Button variant="secondary" size="icon" asChild>
            <Link href="/workspace/agents">
              <ArrowLeft />
            </Link>
          </Button>
          <div className="flex-1 truncate text-sm">{bot.title}</div>
        </div>
      )}

      <SidebarGroupContent>
        <SidebarMenu>
          {_.isEmpty(chats) ? (
            <SidebarMenuItem>
              <SidebarMenuButton
                onClick={chatCreate}
                className="bg-primary text-primary-foreground hover:bg-primary/90 hover:text-primary-foreground active:bg-primary/90 active:text-primary-foreground min-w-8 cursor-pointer duration-200 ease-linear"
              >
                <Plus />
                <span>{sidebar_workspace('chats_new')}</span>
              </SidebarMenuButton>
            </SidebarMenuItem>
          ) : (
            chats?.map((chat) => {
              let url = `/agents/${bot?.id}/chats/${chat.id}`;
              if (workspace) {
                url = '/workspace' + url;
              }
              return (
                <SidebarMenuItem key={chat.id} className="group/item">
                  <SidebarMenuButton
                    asChild
                    isActive={pathname === url}
                    className="data-[active=true]:font-normal"
                  >
                    <Link href={url}>
                      <div className="truncate">
                        {_.isEmpty(chat.title) || chat.title === 'New Chat'
                          ? sidebar_workspace('display_empty_title')
                          : chat.title}
                      </div>
                    </Link>
                  </SidebarMenuButton>
                  <SidebarMenuAction
                    className="invisible cursor-pointer group-hover/item:visible hover:bg-transparent"
                    onClick={() => chatDelete && chatDelete(chat)}
                  >
                    <Trash />
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
