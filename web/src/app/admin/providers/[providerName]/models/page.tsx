import { ModelTable } from '@/app/workspace/providers/[providerName]/models/model-table';
import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { Metadata } from 'next';
import { getTranslations } from 'next-intl/server';

export async function generateMetadata(): Promise<Metadata> {
  const page_models = await getTranslations('page_models');
  return {
    title: page_models('metadata.model_title'),
    description: page_models('metadata.model_description'),
  };
}

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
            href: '/admin/providers',
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
          pathnamePrefix="/admin"
        />
      </PageContent>
    </PageContainer>
  );
}
