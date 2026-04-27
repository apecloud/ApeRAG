import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { getUserPrompts } from '@/features/prompt/server-api';
import { toJson } from '@/lib/utils';
import { getTranslations } from 'next-intl/server';
import { PromptSettings } from './prompt-settings';

export default async function Page() {
  const data = await getUserPrompts();
  const page_prompts = await getTranslations('page_prompts');

  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: page_prompts('metadata.title') }]} />
      <PageContent>
        <PageTitle>{page_prompts('metadata.title')}</PageTitle>
        <PageDescription>{page_prompts('metadata.description')}</PageDescription>
        <PromptSettings data={toJson(data)} />
      </PageContent>
    </PageContainer>
  );
}
