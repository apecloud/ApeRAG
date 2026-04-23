import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import {
  getSettings,
  getSystemDefaultQuotas,
} from '@/features/admin/server-api';
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
  const admin_config = await getTranslations('admin_config');

  const [settings, systemDefaultQuotas] = await Promise.all([
    getSettings(),
    getSystemDefaultQuotas(),
  ]);

  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: admin_config('metadata.title') }]} />
      <PageContent>
        <PageTitle>{admin_config('metadata.title')}</PageTitle>
        <PageDescription className="mb-8">
          {admin_config('metadata.description')}
        </PageDescription>

        <div className="flex flex-col gap-6">
          <ParserSettings data={settings ?? undefined} />
          {systemDefaultQuotas?.quotas ? (
            <QuotaSettings data={systemDefaultQuotas.quotas} />
          ) : null}
        </div>
      </PageContent>
    </PageContainer>
  );
}
