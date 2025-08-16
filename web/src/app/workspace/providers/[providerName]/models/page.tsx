import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';

import { ModelTable } from './model-table';

export default async function Page({
  params,
}: {
  params: Promise<{ providerName: string }>;
}) {
  const { providerName } = await params;
  const serverApi = await getServerApi();

  const [modelsRes, providerRes] = await Promise.all([
    serverApi.defaultApi.llmProvidersProviderNameModelsGet({
      providerName,
    }),
    serverApi.defaultApi.llmProvidersProviderNameGet({
      providerName,
    }),
  ]);

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          { title: 'Providers', href: '/workspace/providers' },
          { title: providerRes.data.label },
          { title: 'Models' },
        ]}
      />
      <PageContent>
        <PageTitle>{providerRes.data.label} Models</PageTitle>
        <PageDescription>
          This section allows you to connect and customize your preferred Large
          Language Model (LLM) providers and models for personal use. Set up API
          keys, choose models, and adjust settings to enhance your AI
          experience.
        </PageDescription>
        <ModelTable
          provider={providerRes.data}
          data={modelsRes.data.items || []}
          pathnamePrefix="/workspace"
        />
      </PageContent>
    </PageContainer>
  );
}
