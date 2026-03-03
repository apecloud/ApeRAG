import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';
import { getTranslations } from 'next-intl/server';
import { PromptSettings } from './prompt-settings';

export default async function Page() {
  const serverApi = await getServerApi();
  const res = await serverApi.defaultApi.promptsUserGet();
  const data = res.data;
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
