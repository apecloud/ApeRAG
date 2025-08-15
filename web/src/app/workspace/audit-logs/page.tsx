import { AuditApiListAuditLogsRequest } from '@/api';
import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';
import { DataTable } from './data-table';

export default async function Page({
  searchParams,
}: {
  searchParams: Promise<AuditApiListAuditLogsRequest>;
}) {
  const serverApi = await getServerApi();

  const defaultEndDate = new Date();
  const defaultStartDate = new Date(
    defaultEndDate.getTime() - 1 * 24 * 60 * 60 * 1000,
  );

  const {
    limit = 200,
    apiName = '',
    startDate = defaultStartDate.toISOString(),
    endDate = defaultEndDate.toISOString(),
  } = await searchParams;
  const res = await serverApi.auditApi.listAuditLogs({
    apiName,
    startDate,
    endDate,
    limit,
  });
  const data = res.data.items || [];

  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: 'Audit Logs' }]} />
      <PageContent>
        <PageTitle>Audit Logs</PageTitle>
        <PageDescription>
          Track and review all critical system activities with Audit Logs—a
          detailed record of user actions, API calls, and administrative
          changes. Ensure transparency, security, and compliance by monitoring
          who did what, when, and from where.
        </PageDescription>
        <DataTable
          data={toJson(data)}
          searchParams={{ limit, apiName, startDate, endDate }}
        />
      </PageContent>
    </PageContainer>
  );
}
