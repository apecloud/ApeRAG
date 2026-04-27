import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';

import { getUserQuota } from '@/features/quota/server-api';
import { getTranslations } from 'next-intl/server';
import { QuotaRadialChart } from './quota-radial-chart';

export default async function Page() {
  const data = await getUserQuota();
  const page_quotas = await getTranslations('page_quota');

  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: page_quotas('metadata.title') }]} />
      <PageContent>
        <PageTitle>{page_quotas('metadata.title')}</PageTitle>
        <PageDescription>{page_quotas('metadata.description')}</PageDescription>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {data.quotas.map((quota) => (
            <QuotaRadialChart key={quota.quota_type} data={quota} />
          ))}
        </div>
      </PageContent>
    </PageContainer>
  );
}
