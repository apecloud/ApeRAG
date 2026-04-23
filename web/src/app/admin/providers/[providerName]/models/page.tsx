import { ModelTable } from '@/app/workspace/providers/[providerName]/models/model-table';
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
            href: '/admin/providers',
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
          pathnamePrefix="/admin"
        />
      </PageContent>
    </PageContainer>
  );
}
