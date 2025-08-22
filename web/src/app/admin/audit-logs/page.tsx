import { AuditApiListAuditLogsRequest } from '@/api';
import { AuditLogTable } from '@/app/workspace/audit-logs/audit-log-table';
import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { parsePageParams, toJson } from '@/lib/utils';

export default async function Page({
  searchParams,
}: {
  searchParams: Promise<AuditApiListAuditLogsRequest>;
}) {
  const serverApi = await getServerApi();

  const {
    page,
    pageSize,
    sortBy = 'created',
    sortOrder = 'desc',
    apiName = '',
    startDate,
    endDate,
    userId,
  } = await searchParams;

  let res;
  try {
    res = await serverApi.auditApi.listAuditLogs({
      apiName,
      sortBy,
      sortOrder,
      startDate,
      endDate,
      userId,
      ...parsePageParams({ page, pageSize }),
    });
  } catch (err) {
    console.log(err);
  }

  //@ts-expect-error api define has a bug
  const data = res?.data?.items || [];

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
          pageCount={res?.data.total_pages || 1}
          urlPrefix="/admin"
        />
      </PageContent>
    </PageContainer>
  );
}
