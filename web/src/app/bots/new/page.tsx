import {
  PageContainer,
  PageContent,
  PageTitle,
} from '@/components/page-container';

import { getDefaultPrompt } from '@/lib/prompt-template';
import { getTranslations } from 'next-intl/server';
import { BotForm } from '../bot-form';
export default async function Page() {
  const page_bot = await getTranslations('page_bot');
  const defaultPrompt = await getDefaultPrompt();
  return (
    <PageContainer>
      <PageContent>
        <PageTitle>{page_bot('new_bot')}</PageTitle>
        <BotForm action="add" defaultPrompt={defaultPrompt} />
      </PageContent>
    </PageContainer>
  );
}
