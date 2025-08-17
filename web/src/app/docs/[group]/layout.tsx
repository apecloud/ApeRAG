import { AppLogo, AppUserDropdownMenu } from '@/components/app-topbar';
import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
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
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
  SidebarProvider,
} from '@/components/ui/sidebar';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { DocsSideBar, getDocsSideBar } from '@/lib/docs';
import { ChevronRight } from 'lucide-react';
import Link from 'next/link';

export default async function Layout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const sidebarData = getDocsSideBar();

  const renderSideBarItem = (
    item: DocsSideBar,
    parentType?: 'folder' | 'group' | 'file',
  ) => {
    let content;

    if (item.type === 'group') {
      content = (
        <SidebarGroup key={item.name}>
          <SidebarGroupLabel>{item.title}</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {item.children?.map((child) => renderSideBarItem(child, 'group'))}
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
          <Tooltip key={item.name}>
            <TooltipTrigger asChild>
              <SidebarMenuSubItem>
                <SidebarMenuSubButton asChild className="truncate">
                  <Link href={item.href || '#'}>{item.title}</Link>
                </SidebarMenuSubButton>
              </SidebarMenuSubItem>
            </TooltipTrigger>
            <TooltipContent side="right">{item.title}</TooltipContent>
          </Tooltip>
        );
      } else if (parentType === 'group') {
        content = (
          <Tooltip key={item.name}>
            <TooltipTrigger asChild>
              <SidebarMenu>
                <SidebarMenuItem>
                  <SidebarMenuButton asChild className="truncate">
                    <Link href={item.href || '#'}>{item.title}</Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              </SidebarMenu>
            </TooltipTrigger>
            <TooltipContent side="right">{item.title}</TooltipContent>
          </Tooltip>
        );
      } else {
        content = (
          <Tooltip key={item.name}>
            <TooltipTrigger asChild>
              <SidebarGroup>
                <SidebarGroupContent>
                  <SidebarMenu>
                    <SidebarMenuItem>
                      <SidebarMenuButton asChild className="truncate">
                        <Link href={item.href || '#'}>{item.title}</Link>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  </SidebarMenu>
                </SidebarGroupContent>
              </SidebarGroup>
            </TooltipTrigger>
            <TooltipContent side="right">{item.title}</TooltipContent>
          </Tooltip>
        );
      }
    }

    return content;
  };

  return (
    <>
      <SidebarProvider>
        <Sidebar>
          <SidebarHeader className="h-16 flex-row items-center gap-4 px-4 align-middle">
            <AppLogo />
          </SidebarHeader>
          <SidebarContent className="gap-0">
            {sidebarData.map((child) => renderSideBarItem(child))}
          </SidebarContent>
          <SidebarFooter className="border-t">
            <AppUserDropdownMenu />
          </SidebarFooter>
        </Sidebar>
        <SidebarInset>
          <PageContainer>
            <PageHeader breadcrumbs={[{ title: 'Documents' }]} />
            <PageContent className="pb-20">{children}</PageContent>
          </PageContainer>
        </SidebarInset>
      </SidebarProvider>
    </>
  );
}
