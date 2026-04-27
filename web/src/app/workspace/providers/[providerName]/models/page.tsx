import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import {
  getProvider,
  getProviderModels,
} from '@/features/providers/server-api';

import { getTranslations } from 'next-intl/server';
import { ModelTable } from './model-table';

export default async function Page({
  params,
}: {
  params: Promise<{ providerName: string }>;
}) {
  const { providerName } = await params;
  const page_models = await getTranslations('page_models');

  const [models, provider] = await Promise.all([
    getProviderModels(providerName),
    getProvider(providerName),
  ]);

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          {
            title: page_models('metadata.provider_title'),
            href: '/workspace/providers',
          },
          { title: provider?.label ?? providerName },
          { title: page_models('metadata.model_title') },
        ]}
      />
      <PageContent>
        <PageTitle>{page_models('metadata.model_title')}</PageTitle>
        <PageDescription>
          {page_models('metadata.model_description')}
        </PageDescription>
        <ModelTable
          provider={provider}
          data={models}
          pathnamePrefix="/workspace"
        />
      </PageContent>
    </PageContainer>
  );
}
