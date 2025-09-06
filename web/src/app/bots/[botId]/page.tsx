import {
  PageContainer,
  PageContent,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { Button } from '@/components/ui/button';
import { getServerApi } from '@/lib/api/server';
import { getTranslations } from 'next-intl/server';
import { notFound } from 'next/navigation';
import { BotForm } from '../bot-form';
import { BotDelete } from './bot-delete';
export default async function Page({
  params,
}: Readonly<{
  children: React.ReactNode;
  params: Promise<{ botId: string }>;
}>) {
  const { botId } = await params;
  const page_bot = await getTranslations('page_bot');
  const serverApi = await getServerApi();
  const botRes = await serverApi.defaultApi.botsBotIdGet({
    botId,
  });

  const bot = botRes.data;

  if (!bot) {
    notFound();
  }

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
        <BotForm bot={bot} action="edit" />
      </PageContent>
    </PageContainer>
  );
}
