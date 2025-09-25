'use client';
import { ChatDetails } from '@/api';
import { ChatMessages } from '@/components/chat/chat-messages';
import { PageContainer, PageContent } from '@/components/page-container';
import { useBotContext } from '@/components/providers/bot-provider';
import _ from 'lodash';
import { useTranslations } from 'next-intl';
import { BotHeader } from '../bot-header';

export const ChatDetail = ({ chat }: { chat: ChatDetails }) => {
  const page_chat = useTranslations('page_chat');
  const page_bot = useTranslations('page_bot');
  const { bot } = useBotContext();
  return (
    <PageContainer>
      <BotHeader
        breadcrumbs={[
          { title: page_bot('shared_bots'), href: `/marketplace/bots` },
          { title: bot?.title || '' },
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
};
