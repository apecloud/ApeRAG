import { AppLogo, AppUserDropdownMenu } from '@/components/app-topbar';
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
  SidebarProvider,
} from '@/components/ui/sidebar';
import { getServerApi } from '@/lib/api/server';
import { BookOpen, DollarSign, Key, Logs, Package, Plus } from 'lucide-react';
import Link from 'next/link';
import { redirect } from 'next/navigation';

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
    redirect('/auth/signin');
  }

  return (
    <>
      <SidebarProvider>
        <Sidebar variant="inset">
          <SidebarHeader className="flex-row gap-4">
            <AppLogo />
          </SidebarHeader>
          <SidebarContent>
            <SidebarGroup className="flex-auto">
              <SidebarGroupLabel>Chats</SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  <SidebarMenuItem className="flex items-center gap-2">
                    <SidebarMenuButton
                      tooltip="Quick Create"
                      className="bg-primary text-primary-foreground hover:bg-primary/90 hover:text-primary-foreground active:bg-primary/90 active:text-primary-foreground min-w-8 duration-200 ease-linear"
                    >
                      <Plus />
                      <span>Create chat</span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>

            <SidebarGroup>
              <SidebarGroupLabel>Settings</SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  <SidebarMenuItem className="flex items-center gap-2">
                    <SidebarMenuButton asChild>
                      <Link href="/workspace/collections">
                        <BookOpen />
                        <span>Collections</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                  <SidebarMenuItem>
                    <SidebarMenuButton asChild>
                      <Link href="/workspace/models">
                        <Package /> Models
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                  <SidebarMenuItem>
                    <SidebarMenuButton asChild>
                      <Link href="/workspace/api-keys">
                        <Key /> API Keys
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                  <SidebarMenuItem>
                    <SidebarMenuButton asChild>
                      <Link href="/workspace/audit-logs">
                        <Logs /> Audit Logs
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                  <SidebarMenuItem>
                    <SidebarMenuButton asChild>
                      <Link href="/workspace/quota">
                        <DollarSign /> Quota
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          </SidebarContent>
          <SidebarFooter>
            <AppUserDropdownMenu />
          </SidebarFooter>
        </Sidebar>
        <SidebarInset>{children}</SidebarInset>
      </SidebarProvider>
    </>
  );
}
