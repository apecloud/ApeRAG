import { ChatMessages } from '@/components/chat/chat-messages';
import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { getBotChat } from '@/features/bot/server-api';
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
  const page_chat = await getTranslations('page_chat');

  let chat;

  try {
    chat = await getBotChat(botId, chatId);
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
  } catch (err) {
    notFound();
  }
  if (!chat) {
    notFound();
  }

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          {
            title:
              page_chat('metadata.title') +
              ': ' +
              (_.isEmpty(chat.history)
                ? page_chat('display_empty_title')
                : chat.title || ''),
          },
        ]}
      />
      <PageContent>
        <ChatMessages chat={chat} />
      </PageContent>
    </PageContainer>
  );
}
