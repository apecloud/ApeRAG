'use client';

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import {
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
} from '@/components/ui/sidebar';
import { DocsSideBar } from '@/lib/docs';

import { ChevronRight } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useCallback } from 'react';

export const DocsSideBarItem = ({ child }: { child: DocsSideBar }) => {
  const pathname = usePathname();
  const renderSideBarItem = useCallback(
    (item: DocsSideBar, parentType?: 'folder' | 'group' | 'file') => {
      let content;

      if (item.type === 'group') {
        content = (
          <SidebarGroup key={item.name}>
            <SidebarGroupLabel>{item.title}</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {item.children?.map((child) =>
                  renderSideBarItem(child, 'group'),
                )}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        );
      }

      if (item.type === 'folder') {
        content = (
          <Collapsible key={item.name} asChild className="group/collapsible">
            <SidebarMenuItem>
              <CollapsibleTrigger asChild>
                <SidebarMenuButton>
                  {item.title}
                  <ChevronRight className="ml-auto transition-transform duration-200 group-data-[state=open]/collapsible:rotate-90" />
                </SidebarMenuButton>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <SidebarMenuSub>
                  {item.children?.map((child) =>
                    renderSideBarItem(child, 'folder'),
                  )}
                </SidebarMenuSub>
              </CollapsibleContent>
            </SidebarMenuItem>
          </Collapsible>
        );
      }

      if (item.type === 'file') {
        if (parentType === 'folder') {
          content = (
            <SidebarMenuSubItem key={item.name}>
              <SidebarMenuSubButton
                asChild
                isActive={pathname === item.href}
                className="data-[active=true]:font-normal"
              >
                <Link href={item.href || '#'}>
                  <div className="truncate">{item.title}</div>
                </Link>
              </SidebarMenuSubButton>
            </SidebarMenuSubItem>
          );
        } else if (parentType === 'group') {
          content = (
            <SidebarMenu key={item.name}>
              <SidebarMenuItem>
                <SidebarMenuButton
                  asChild
                  isActive={pathname === item.href}
                  className="data-[active=true]:font-normal"
                >
                  <Link href={item.href || '#'}>
                    <div className="truncate">{item.title}</div>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          );
        } else {
          content = (
            <SidebarGroup key={item.name}>
              <SidebarGroupContent>
                <SidebarMenu>
                  <SidebarMenuItem>
                    <SidebarMenuButton
                      asChild
                      className="data-[active=true]:font-normal"
                      isActive={pathname === item.href}
                    >
                      <Link href={item.href || '#'}>
                        <div className="truncate">{item.title}</div>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          );
        }
      }

      return content;
    },
    [pathname],
  );

  return renderSideBarItem(child);
};
