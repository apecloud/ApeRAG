'use client';

import { useWorkspaceContext } from '@/components/providers/workspace-provider';
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
import { motion } from 'framer-motion';
import _ from 'lodash';
import { Plus, Trash } from 'lucide-react';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

export const MenuChats = () => {
  const { bot, chats, chatCreate, chatDelete } = useWorkspaceContext();
  const pathname = usePathname();
  const sidebar_workspace = useTranslations('sidebar_workspace');
  return (
    <SidebarGroup>
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
              const url = `/workspace/agents/${bot?.id}/chats/${chat.id}`;
              return (
                <motion.div
                  key={chat.id}
                  // initial={{ opacity: 0, x: 40 }}
                  // animate={{ opacity: 1, x: 0 }}
                  // transition={{
                  //   duration: 0.2,
                  //   ease: 'easeIn',
                  //   delay: index * 0.04,
                  // }}
                >
                  <SidebarMenuItem className="group/item">
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
                </motion.div>
              );
            })
          )}
        </SidebarMenu>
      </SidebarGroupContent>
    </SidebarGroup>
  );
};
