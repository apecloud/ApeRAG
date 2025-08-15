import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';

import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';
import { ApiKeyTable } from './api-key-table';

export default async function Page() {
  const serverApi = await getServerApi();
  const res = await serverApi.defaultApi.apikeysGet();
  const data = res.data.items || [];

  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: 'API keys' }]} />
      <PageContent>
        <PageTitle>API keys</PageTitle>
        <PageDescription>
          This section allows you to securely store, update, and manage API keys
          for different AI services (such as OpenAI, Anthropic, Gemini, etc.).
          Connecting your keys enables personalized AI interactions while
          keeping your credentials protected.
        </PageDescription>
        <ApiKeyTable data={toJson(data)} />
      </PageContent>
    </PageContainer>
  );
}
