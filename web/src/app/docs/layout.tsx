import { AppLogo, AppUserDropdownMenu } from '@/components/app-topbar';
import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarInset,
  SidebarProvider,
} from '@/components/ui/sidebar';
import { getDocsSideBar } from '@/lib/docs';
import { DocsSideBarItem } from './docs-sidebar';

export default async function Layout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const sidebarData = getDocsSideBar();

  return (
    <>
      <SidebarProvider>
        <Sidebar>
          <SidebarHeader className="h-16 flex-row items-center gap-4 px-4 align-middle">
            <AppLogo />
          </SidebarHeader>
          <SidebarContent className="gap-0">
            {sidebarData.map((child) => (
              <DocsSideBarItem key={child.name} child={child} />
            ))}
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
