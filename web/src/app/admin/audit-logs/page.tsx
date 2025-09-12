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
import { Metadata } from 'next';
import { getTranslations } from 'next-intl/server';

export async function generateMetadata(): Promise<Metadata> {
  const page_audit_logs = await getTranslations('page_audit_logs');
  return {
    title: page_audit_logs('metadata.title'),
    description: page_audit_logs('metadata.description'),
  };
}

export default async function Page({
  searchParams,
}: {
  searchParams: Promise<AuditApiListAuditLogsRequest>;
}) {
  const page_audit_logs = await getTranslations('page_audit_logs');
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
      <PageHeader
        breadcrumbs={[{ title: page_audit_logs('metadata.title') }]}
      />
      <PageContent>
        <PageTitle>{page_audit_logs('metadata.title')}</PageTitle>
        <PageDescription>
          {page_audit_logs('metadata.description')}
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
