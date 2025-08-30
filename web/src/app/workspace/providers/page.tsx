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
import { ProviderTable } from './provider-table';

export default async function Page() {
  const serverApi = await getServerApi();
  const page_models = await getTranslations('page_models');
  const res = await serverApi.defaultApi.llmConfigurationGet();

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[{ title: page_models('metadata.provider_title') }]}
      />
      <PageContent>
        <PageTitle>{page_models('metadata.provider_title')}</PageTitle>
        <PageDescription>
          {page_models('metadata.provider_description')}
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
