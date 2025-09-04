import { Chat } from '@/api';
import { getServerApi } from '@/lib/api/server';
import _ from 'lodash';
import { redirect } from 'next/navigation';

export default async function Page({
  params,
}: Readonly<{
  params: Promise<{ botId: string }>;
}>) {
  const apiServer = await getServerApi();

  const { botId } = await params;
  const chatsRes = await apiServer.defaultApi.botsBotIdChatsGet({
    botId,
    page: 1,
    pageSize: 1,
  });

  //@ts-expect-error api define has a bug
  const chat: Chat | undefined = _.first(chatsRes.data.items || []);

  if (chat) {
    redirect(`/bots/${botId}/chats/${chat.id}`);
  } else {
    const res = await apiServer.defaultApi.botsBotIdChatsPost({
      botId,
      chatCreate: {
        title: '',
      },
    });
    if (res.data.id) {
      redirect(`/bots/${botId}/chats/${res.data.id}`);
    }
  }

  return '';
}
