import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';
import { notFound } from 'next/navigation';
import { ChatDetail } from './chat-detail';

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

  return <ChatDetail chat={toJson(chat)} />;
}
