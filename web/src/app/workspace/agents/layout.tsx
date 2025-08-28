import { AgentsProvider } from '@/components/providers/agents-provider';

export default async function ChatLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return <AgentsProvider>{children}</AgentsProvider>;
}
