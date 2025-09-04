import {
  PageContainer,
  PageContent,
  PageDescription,
  PageTitle,
} from '@/components/page-container';

import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';
import { getTranslations } from 'next-intl/server';
import { BotList } from './bot-list';

export default async function Page() {
  const page_bot = await getTranslations('page_bot');
  const serverApi = await getServerApi();
  const res = await serverApi.defaultApi.botsGet();
  const bots = res.data.items || [];

  return (
    <PageContainer>
      <PageContent>
        <PageTitle>{page_bot('metadata.title')}</PageTitle>
        <PageDescription>{page_bot('metadata.description')}</PageDescription>
        <BotList bots={toJson(bots)} />
      </PageContent>
    </PageContainer>
  );
}
