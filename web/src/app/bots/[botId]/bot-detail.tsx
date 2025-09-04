'use client';

import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { useBotContext } from '@/components/providers/bot-provider';
import { useTranslations } from 'next-intl';

export const BotDetail = () => {
  const page_bot = useTranslations('page_bot');
  const { bot } = useBotContext();
  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          { title: page_bot('metadata.title'), href: `/bots` },
          { title: bot.title || '' },
        ]}
        extra=""
      />
      <PageContent></PageContent>
    </PageContainer>
  );
};
