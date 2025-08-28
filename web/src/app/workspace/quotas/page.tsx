import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';

import { UserQuotaInfo } from '@/api';
import { getServerApi } from '@/lib/api/server';
import { QuotaRadialChart } from './quota-radial-chart';

export default async function Page() {
  const serverApi = await getServerApi();
  const res = await serverApi.quotasApi.quotasGet();

  const data = res.data as UserQuotaInfo;

  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: 'Quotas' }]} />
      <PageContent>
        <PageTitle>Quotas</PageTitle>
        <PageDescription>
          Manage your usage limits for AI models, API calls, and other resources
          in this section. Quotas help you control costs, prevent overuse, and
          optimize your experience.
        </PageDescription>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {data.quotas.map((quota) => (
            <QuotaRadialChart key={quota.quota_type} data={quota} />
          ))}
        </div>
      </PageContent>
    </PageContainer>
  );
}
