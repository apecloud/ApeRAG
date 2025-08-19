import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
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

  const res = await serverApi.defaultApi.botsBotIdChatsChatIdGet({
    botId,
    chatId,
  });

  const chat = res.data;

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
