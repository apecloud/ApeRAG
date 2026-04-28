import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { listBotChats } from '@/features/bot/server-api';
import type { Chat } from '@/features/bot/types';
import _ from 'lodash';
import { getTranslations } from 'next-intl/server';
import { redirect } from 'next/navigation';

export default async function Page({
  params,
}: Readonly<{
  params: Promise<{ botId: string }>;
}>) {
  const page_chat = await getTranslations('page_chat');
  const { botId } = await params;
  const chatsRes = await listBotChats(botId, { page: 1, pageSize: 1 });
  const chat: Chat | undefined = _.first(chatsRes.items ?? []);

  if (chat) {
    redirect(`/workspace/bots/${botId}/chats/${chat.id}`);
  }

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[{ title: page_chat('metadata.title') }]}
        extra=""
      />
      <PageContent></PageContent>
    </PageContainer>
  );
}
