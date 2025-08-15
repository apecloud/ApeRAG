import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';
import { ProviderTable } from './provider-table';

export default async function Page() {
  const serverApi = await getServerApi();

  const res = await serverApi.defaultApi.llmConfigurationGet();

  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: 'Providers' }]} />
      <PageContent>
        <PageTitle>Model Provider</PageTitle>
        <PageDescription>
          This section allows you to connect and customize your preferred Large
          Language Model (LLM) providers and models for personal use. Set up API
          keys, choose models, and adjust settings to enhance your AI
          experience.
        </PageDescription>
        <ProviderTable
          data={toJson(res.data.providers) || []}
          models={toJson(res.data.models) || []}
          urlPrefix="/workspace"
        />
      </PageContent>
    </PageContainer>
  );
}
