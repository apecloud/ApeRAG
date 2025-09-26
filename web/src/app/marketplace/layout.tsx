import { getServerApi } from '@/lib/api/server';
import { Metadata } from 'next';
import { getTranslations } from 'next-intl/server';
import { redirect } from 'next/navigation';

export async function generateMetadata(): Promise<Metadata> {
  const page_marketplace = await getTranslations('page_marketplace');
  return {
    title: page_marketplace('metadata.title'),
    description: page_marketplace('metadata.description'),
  };
}

export default async function RootLayout({
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
      `/auth/signin?callbackUrl=${encodeURIComponent(`${process.env.NEXT_PUBLIC_BASE_PATH}/marketplace/bots`)}`,
    );
  }
  return <>{children}</>;
}
