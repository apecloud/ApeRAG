import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { listAuditLogs } from '@/features/audit/server-api';
import type { ListAuditLogsParams } from '@/features/audit/types';
import { parsePageParams, toJson } from '@/lib/utils';
import { getTranslations } from 'next-intl/server';
import { AuditLogTable } from './audit-log-table';

export default async function Page({
  searchParams,
}: {
  searchParams: Promise<ListAuditLogsParams>;
}) {
  const page_audit_logs = await getTranslations('page_audit_logs');
  const {
    page,
    pageSize,
    sortBy = 'created',
    sortOrder = 'desc',
    apiName = '',
    startDate,
    endDate,
  } = await searchParams;

  let res;
  try {
    res = await listAuditLogs({
      apiName,
      sortBy,
      sortOrder,
      startDate,
      endDate,
      ...parsePageParams({ page, pageSize }),
    });
  } catch (err) {
    console.log(err);
  }

  const data = res?.items || [];

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
          pageCount={res?.total_pages || 1}
          urlPrefix="/workspace"
        />
      </PageContent>
    </PageContainer>
  );
}
