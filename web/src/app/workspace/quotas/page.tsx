import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';

import { UserQuotaInfo } from '@/api';
import { getServerApi } from '@/lib/api/server';
import { getTranslations } from 'next-intl/server';
import { QuotaRadialChart } from './quota-radial-chart';

export default async function Page() {
  const serverApi = await getServerApi();
  const res = await serverApi.quotasApi.quotasGet();
  const page_quotas = await getTranslations('page_quota');
  const data = res.data as UserQuotaInfo;

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
