import {
  PageContainer,
  PageContent,
  PageDescription,
  PageTitle,
} from '@/components/page-container';

import { Bot, SharedBot } from '@/api';
import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';
import { getTranslations } from 'next-intl/server';
import { BotList } from './bot-list';

export default async function Page() {
  const page_bot = await getTranslations('page_bot');
  const serverApi = await getServerApi();

  let bots: Bot[] = [];
  let sharedBots: SharedBot[] = [];

  try {
    const [botsRes, marketplaceBotsRes] = await Promise.all([
      serverApi.defaultApi.botsGet(),
      serverApi.defaultApi.marketplaceBotsGet(),
    ]);
    bots = botsRes.data.items || [];
    sharedBots = marketplaceBotsRes.data.items || [];
  } catch (err) {
    console.log(err);
  }

  return (
    <PageContainer>
      <PageContent>
        <PageTitle>{page_bot('metadata.title')}</PageTitle>
        <PageDescription>{page_bot('metadata.description')}</PageDescription>
        <BotList bots={toJson(bots)} sharedBots={toJson(sharedBots)} />
      </PageContent>
    </PageContainer>
  );
}
