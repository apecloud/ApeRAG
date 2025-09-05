'use client';

import {
  PageContainer,
  PageContent,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { useBotContext } from '@/components/providers/bot-provider';
import { Button } from '@/components/ui/button';
import { useTranslations } from 'next-intl';
import { useRouter } from 'next/navigation';
import { BotForm } from '../bot-form';
import { BotDelete } from './bot-delete';

export const BotSettings = () => {
  const page_bot = useTranslations('page_bot');
  const { bot, botReload } = useBotContext();
  const router = useRouter();
  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          { title: page_bot('metadata.title'), href: `/bots` },
          { title: bot?.title || '' },
        ]}
        extra=""
      />
      <PageContent>
        <div className="flex flex-row justify-between">
          <PageTitle>{page_bot('bot_settings')}</PageTitle>

          <BotDelete>
            <Button
              variant="ghost"
              className="cursor-pointer text-red-400 hover:text-red-500"
            >
              {page_bot('delete_bot')}
            </Button>
          </BotDelete>
        </div>

        <BotForm
          bot={bot}
          action="edit"
          onUpdateSuccess={(value) => {
            router.push(`/bots/${value.id}/chats`);
            if (botReload) {
              botReload();
            }
          }}
        />
      </PageContent>
    </PageContainer>
  );
};
