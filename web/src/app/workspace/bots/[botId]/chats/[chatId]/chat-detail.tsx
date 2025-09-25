'use client';
import { ChatDetails } from '@/api';
import { ChatMessages } from '@/components/chat/chat-messages';
import { PageContainer, PageContent } from '@/components/page-container';
import { useBotContext } from '@/components/providers/bot-provider';
import { Button } from '@/components/ui/button';
import _ from 'lodash';
import { Settings } from 'lucide-react';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { BotHeader } from '../bot-header';

export const ChatDetail = ({ chat }: { chat: ChatDetails }) => {
  const page_chat = useTranslations('page_chat');
  const page_bot = useTranslations('page_bot');
  const { bot } = useBotContext();
  return (
    <PageContainer>
      <BotHeader
        breadcrumbs={[
          { title: page_bot('metadata.title'), href: `/workspace/bots` },
          { title: bot?.title || '', href: `/workspace/bots/${bot?.id}` },
          {
            title:
              page_chat('metadata.title') +
              ': ' +
              (_.isEmpty(chat.history)
                ? page_chat('display_empty_title')
                : chat.title || ''),
          },
        ]}
        extra={
          <Button size="icon" variant="ghost">
            <Link href={`/workspace/bots/${bot?.id}`}>
              <Settings />
            </Link>
          </Button>
        }
      />
      <PageContent>
        <ChatMessages chat={chat} />
      </PageContent>
    </PageContainer>
  );
};
