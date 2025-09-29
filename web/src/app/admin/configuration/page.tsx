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
import { ParserSettings } from './parser-settings';
import { QuotaSettings } from './quota-settings';

export async function generateMetadata(): Promise<Metadata> {
  const admin_config = await getTranslations('admin_config');
  return {
    title: admin_config('metadata.title'),
    description: admin_config('metadata.description'),
  };
}

export default async function Page() {
  const serverApi = await getServerApi();
  const admin_config = await getTranslations('admin_config');

  const [resSettings, resSystemDefaultQuotas] = await Promise.all([
    serverApi.defaultApi.settingsGet(),
    serverApi.quotasApi.systemDefaultQuotasGet(),
  ]);

  const settings = resSettings.data;

  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: admin_config('metadata.title') }]} />
      <PageContent>
        <PageTitle>{admin_config('metadata.title')}</PageTitle>
        <PageDescription className="mb-8">
          {admin_config('metadata.description')}
        </PageDescription>

        <div className="flex flex-col gap-6">
          <ParserSettings data={settings} />
          <QuotaSettings data={resSystemDefaultQuotas.data.quotas} />
        </div>
      </PageContent>
    </PageContainer>
  );
}
