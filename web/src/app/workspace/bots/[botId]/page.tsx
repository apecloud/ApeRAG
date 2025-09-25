import {
  PageContainer,
  PageContent,
  PageTitle,
} from '@/components/page-container';
import { Button } from '@/components/ui/button';
import { getServerApi } from '@/lib/api/server';
import { getDefaultPrompt } from '@/lib/prompt-template';
import { Star } from 'lucide-react';
import { getTranslations } from 'next-intl/server';
import { notFound } from 'next/navigation';
import { BotForm } from '../bot-form';
import { BotDelete } from './bot-delete';
import { BotSharing } from './bot-sharing';
import { BotHeader } from './chats/bot-header';

export default async function Page({
  params,
}: Readonly<{
  params: Promise<{ botId: string }>;
}>) {
  const { botId } = await params;
  const page_bot = await getTranslations('page_bot');
  const serverApi = await getServerApi();
  const botRes = await serverApi.defaultApi.botsBotIdGet({
    botId,
  });

  const bot = botRes.data;

  const defaultPrompt = await getDefaultPrompt();

  if (!bot) {
    notFound();
  }

  return (
    <PageContainer>
      <BotHeader
        breadcrumbs={[
          { title: page_bot('metadata.title'), href: `/workspace/bots` },
          { title: bot?.title || '' },
        ]}
        extra=""
      />
      <PageContent>
        <div className="flex flex-row justify-between">
          <PageTitle>{page_bot('bot_settings')}</PageTitle>
          <div className="flex flex-row gap-4">
            <BotDelete>
              <Button
                variant="ghost"
                className="cursor-pointer text-red-400 hover:text-red-500"
              >
                {page_bot('delete_bot')}
              </Button>
            </BotDelete>
            <BotSharing>
              <Button variant="ghost" className="cursor-pointer">
                <Star />
                {page_bot('share_settings')}
                {/* <StarOff /> */}
              </Button>
            </BotSharing>
          </div>
        </div>
        <BotForm bot={bot} action="edit" defaultPrompt={defaultPrompt} />
      </PageContent>
    </PageContainer>
  );
}
