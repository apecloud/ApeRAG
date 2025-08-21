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
import { AuditLogTable } from './audit-log-table';

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
    pageSize = 20,
    apiName = '',
    startDate = defaultStartDate.toISOString(),
    endDate = defaultEndDate.toISOString(),
  } = await searchParams;

  let data = [];
  try {
    const res = await serverApi.auditApi.listAuditLogs({
      apiName,
      startDate,
      endDate,
      pageSize,
    });
    //@ts-expect-error api define has a bug
    data = res.data.items || [];
  } catch (err) {
    console.log(err);
  }

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
        <AuditLogTable
          data={toJson(data)}
          searchParams={{ pageSize, apiName, startDate, endDate }}
        />
      </PageContent>
    </PageContainer>
  );
}
