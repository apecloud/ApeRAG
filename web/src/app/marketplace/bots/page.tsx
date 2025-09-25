import {
  PageContainer,
  PageContent,
  PageDescription,
  PageTitle,
} from '@/components/page-container';

import { Bot } from '@/api';
import { Button } from '@/components/ui/button';
import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';
import { getTranslations } from 'next-intl/server';
import Link from 'next/link';
import { BotList } from './bot-list';

export default async function Page() {
  const page_bot = await getTranslations('page_bot');
  const serverApi = await getServerApi();

  let bots: Bot[] = [];
  try {
    const [botsRes] = await Promise.all([
      serverApi.defaultApi.marketplaceBotsGet(),
    ]);
    bots = botsRes.data.items || [];
  } catch (err) {
    console.log(err);
  }

  return (
    <PageContainer>
      <PageContent>
        <div className="flex flex-row items-center justify-between">
          <PageTitle>{page_bot('shared_bots')}</PageTitle>
          <Button variant="secondary" asChild>
            <Link href="/workspace/bots">{page_bot('metadata.title')}</Link>
          </Button>
        </div>

        <PageDescription>{page_bot('metadata.description')}</PageDescription>
        <BotList bots={toJson(bots)} />
      </PageContent>
    </PageContainer>
  );
}
