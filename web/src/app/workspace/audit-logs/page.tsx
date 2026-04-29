import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { Card, CardContent } from '@/components/ui/card';
import { listAuditLogs } from '@/features/audit/server-api';
import type { ListAuditLogsParams } from '@/features/audit/types';
import { parsePageParams, toJson } from '@/lib/utils';
import { Activity, CheckCircle2, ScrollText, XCircle } from 'lucide-react';
import { getTranslations } from 'next-intl/server';
import type { ReactNode } from 'react';
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
  const successCount = data.filter(
    (item) => (item.status_code || 0) >= 200 && (item.status_code || 0) < 400,
  ).length;
  const errorCount = data.filter(
    (item) => (item.status_code || 0) >= 400,
  ).length;
  const averageDuration =
    data.length === 0
      ? 0
      : Math.round(
          data.reduce((sum, item) => sum + (item.duration_ms || 0), 0) /
            data.length,
        );

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[{ title: page_audit_logs('metadata.title') }]}
      />
      <PageContent className="max-w-7xl px-5 py-8 md:px-8 md:py-10">
        <div className="mb-8 flex flex-col gap-5 lg:flex-row lg:items-end">
          <div className="min-w-0 flex-1">
            <div className="text-muted-foreground font-mono text-[11px] uppercase tracking-[0.12em]">
              {page_audit_logs('metadata.label')}
            </div>
            <h1 className="mt-2 font-serif text-4xl font-normal leading-none tracking-normal md:text-[44px]">
              {page_audit_logs('metadata.title')}
            </h1>
            <p className="text-muted-foreground mt-3 max-w-2xl text-sm leading-6">
              {page_audit_logs('metadata.description')}
            </p>
          </div>
        </div>
        <div className="mb-6 grid grid-cols-2 gap-3 md:grid-cols-4">
          <AuditMetric
            icon={<ScrollText className="size-4" />}
            label={page_audit_logs('metric_events')}
            value={data.length}
          />
          <AuditMetric
            icon={<CheckCircle2 className="size-4" />}
            label={page_audit_logs('metric_success')}
            value={successCount}
          />
          <AuditMetric
            icon={<XCircle className="size-4" />}
            label={page_audit_logs('metric_errors')}
            value={errorCount}
          />
          <AuditMetric
            icon={<Activity className="size-4" />}
            label={page_audit_logs('metric_avg_duration')}
            value={averageDuration}
            suffix="ms"
          />
        </div>
        <AuditLogTable
          data={toJson(data)}
          pageCount={res?.total_pages || 1}
          urlPrefix="/workspace"
        />
      </PageContent>
    </PageContainer>
  );
}

const AuditMetric = ({
  icon,
  label,
  value,
  suffix,
}: {
  icon: ReactNode;
  label: string;
  value: number;
  suffix?: string;
}) => {
  return (
    <Card className="border-border/70 gap-0 rounded-xl py-0">
      <CardContent className="flex items-center gap-3 p-4">
        <div className="bg-accent-soft text-accent-ink flex size-9 items-center justify-center rounded-lg">
          {icon}
        </div>
        <div>
          <div className="font-mono text-xl tabular-nums leading-none">
            {value}
            {suffix ? (
              <span className="text-muted-foreground ml-1 text-xs">
                {suffix}
              </span>
            ) : null}
          </div>
          <div className="text-muted-foreground mt-1 text-xs">{label}</div>
        </div>
      </CardContent>
    </Card>
  );
};
