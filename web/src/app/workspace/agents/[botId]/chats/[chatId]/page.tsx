import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { notFound } from 'next/navigation';
import { ChatMessages } from './chat-messages';

export default async function Page({
  params,
}: {
  params: Promise<{
    botId: string;
    chatId: string;
  }>;
}) {
  const { botId, chatId } = await params;
  const serverApi = await getServerApi();

  let chat;

  try {
    const res = await serverApi.defaultApi.botsBotIdChatsChatIdGet({
      botId,
      chatId,
    });
    chat = res.data;
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
  } catch (err) {
    notFound();
  }

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[{ title: 'Chats' }, { title: chat.title || '' }]}
      />
      <PageContent>
        <ChatMessages chat={chat} />
      </PageContent>
    </PageContainer>
  );
}
