import { AppLogo, AppUserDropdownMenu } from '@/components/app-topbar';
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarInset,
  SidebarProvider,
} from '@/components/ui/sidebar';
import { getServerApi } from '@/lib/api/server';

import { notFound, redirect } from 'next/navigation';
import { AdminSideBarMenu } from './admin-sidebar-menu';

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
    redirect(`/auth/signin?callbackUrl=${encodeURIComponent('/admin')}`);
  }

  if (user.role !== 'admin') {
    notFound();
  }

  return (
    <SidebarProvider>
      <Sidebar>
        <SidebarHeader className="h-16 flex-row items-center gap-4 px-4 align-middle">
          <AppLogo />
        </SidebarHeader>
        <SidebarContent className="gap-0">
          {/* <SidebarGroup>
              <SidebarGroupContent>
                <SidebarMenu>
                  <SidebarMenuItem>
                    <SidebarMenuButton asChild isActive>
                      <Link href="/workspace">
                        <ArrowLeft /> My Workspace
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup> */}

          <AdminSideBarMenu />
        </SidebarContent>

        <SidebarFooter className="border-t">
          <AppUserDropdownMenu />
        </SidebarFooter>
      </Sidebar>
      <SidebarInset>{children}</SidebarInset>
    </SidebarProvider>
  );
}
