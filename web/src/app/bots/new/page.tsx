import {
  PageContainer,
  PageContent,
  PageTitle,
} from '@/components/page-container';

import { getTranslations } from 'next-intl/server';
import { BotForm } from '../bot-form';
export default async function Page() {
  const page_bot = await getTranslations('page_bot');

  return (
    <PageContainer>
      <PageContent>
        <PageTitle>{page_bot('new_bot')}</PageTitle>
        <BotForm action="add" />
      </PageContent>
    </PageContainer>
  );
}
