import { ChatMessages } from '@/components/chat/chat-messages';
import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import _ from 'lodash';
import { getTranslations } from 'next-intl/server';
import { notFound } from 'next/navigation';

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
  const page_chat = await getTranslations('page_chat');
  const page_bot = await getTranslations('page_bot');

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
        breadcrumbs={[
          { title: page_bot('metadata.title'), href: `/workspace/agents` },
          {
            title:
              page_chat('metadata.title') +
              ': ' +
              (_.isEmpty(chat.history)
                ? page_chat('display_empty_title')
                : chat.title || ''),
          },
        ]}
        extra=""
      />
      <PageContent>
        <ChatMessages chat={chat} />
      </PageContent>
    </PageContainer>
  );
}
