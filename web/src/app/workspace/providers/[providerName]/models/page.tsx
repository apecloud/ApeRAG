import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';

import { getTranslations } from 'next-intl/server';
import { ModelTable } from './model-table';

export default async function Page({
  params,
}: {
  params: Promise<{ providerName: string }>;
}) {
  const { providerName } = await params;
  const serverApi = await getServerApi();
  const page_models = await getTranslations('page_models');

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
          {
            title: page_models('metadata.provider_title'),
            href: '/workspace/providers',
          },
          { title: providerRes.data.label },
          { title: page_models('metadata.model_title') },
        ]}
      />
      <PageContent>
        <PageTitle>{page_models('metadata.model_title')}</PageTitle>
        <PageDescription>
          {page_models('metadata.model_description')}
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
