import { SidebarInset, SidebarProvider } from '@/components/ui/sidebar';
import { getServerApi } from '@/lib/api/server';

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
    redirect(
      `/auth/signin?callbackUrl=${encodeURIComponent(`${process.env.NEXT_PUBLIC_BASE_PATH}/workspace/bots`)}`,
    );
  }

  return (
    <SidebarProvider>
      <div />
      <SidebarInset>{children}</SidebarInset>
    </SidebarProvider>
  );
}
