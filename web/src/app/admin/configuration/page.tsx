import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { MinerUSettings } from './mineru-settings';
import { QuotaSettings } from './quota-settings';

export default async function Page() {
  const serverApi = await getServerApi();

  const [resSettings, resSystemDefaultQuotas] = await Promise.all([
    serverApi.defaultApi.settingsGet(),
    serverApi.quotasApi.systemDefaultQuotasGet(),
  ]);

  const settings = resSettings.data;

  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: 'Configuration' }]} />
      <PageContent>
        <PageTitle>Configuration</PageTitle>
        <PageDescription className="mb-8">
          This section allows administrators to define and manage system-wide
          settings that apply across the entire platform. These global
          parameters control core functionalities, security policies,
          performance thresholds, and other critical configurations.
        </PageDescription>

        <div className="flex flex-col gap-6">
          <MinerUSettings data={settings} />
          <QuotaSettings data={resSystemDefaultQuotas.data.quotas} />
        </div>
      </PageContent>
    </PageContainer>
  );
}
