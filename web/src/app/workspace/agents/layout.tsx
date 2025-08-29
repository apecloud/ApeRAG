import { AgentsProvider } from '@/components/providers/agents-provider';
import { Metadata } from 'next';
import { getTranslations } from 'next-intl/server';

export async function generateMetadata(): Promise<Metadata> {
  const page_chat = await getTranslations('page_chat');
  return {
    title: page_chat('metadata.title'),
  };
}

export default async function ChatLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return <AgentsProvider>{children}</AgentsProvider>;
}
