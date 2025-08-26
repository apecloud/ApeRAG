import { ProviderTable } from '@/app/workspace/providers/provider-table';
import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';

export default async function Page() {
  const serverApi = await getServerApi();

  const res = await serverApi.defaultApi.llmConfigurationGet();

  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: 'Models' }]} />
      <PageContent>
        <PageTitle>Models & Providers</PageTitle>
        <PageDescription>
          This section allows administrators to manage and integrate third-party
          Large Language Model (LLM) providers and their respective models into
          the system. Configure API keys, model selection, rate limits, and
          other parameters to customize AI-powered functionalities.
        </PageDescription>

        <ProviderTable
          data={toJson(res.data.providers) || []}
          models={toJson(res.data.models) || []}
          urlPrefix="/admin"
        />
      </PageContent>
    </PageContainer>
  );
}
