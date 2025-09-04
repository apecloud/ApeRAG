import { Chat } from '@/api';
import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import _ from 'lodash';
import { getTranslations } from 'next-intl/server';
import { redirect } from 'next/navigation';

export default async function Page({
  params,
}: Readonly<{
  params: Promise<{ botId: string }>;
}>) {
  const apiServer = await getServerApi();
  const page_chat = await getTranslations('page_chat');
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
