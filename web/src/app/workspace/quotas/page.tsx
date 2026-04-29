import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { Card, CardContent } from '@/components/ui/card';

import { getUserQuota } from '@/features/quota/server-api';
import { Gauge, ShieldCheck, TimerReset } from 'lucide-react';
import { getTranslations } from 'next-intl/server';
import type { ReactNode } from 'react';
import { QuotaChartGrid } from './quota-chart-grid';

export default async function Page() {
  const data = await getUserQuota();
  const page_quotas = await getTranslations('page_quota');
  const totalUsage = data.quotas.reduce(
    (sum, quota) => sum + (quota.current_usage || 0),
    0,
  );
  const totalLimit = data.quotas.reduce(
    (sum, quota) => sum + (quota.quota_limit || 0),
    0,
  );

  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: page_quotas('metadata.title') }]} />
      <PageContent className="max-w-7xl px-5 py-8 md:px-8 md:py-10">
        <div className="mb-8 flex flex-col gap-5 lg:flex-row lg:items-end">
          <div className="min-w-0 flex-1">
            <div className="text-muted-foreground font-mono text-[11px] tracking-[0.12em] uppercase">
              {page_quotas('metadata.label')}
            </div>
            <h1 className="mt-2 font-serif text-4xl leading-none font-normal tracking-normal md:text-[44px]">
              {page_quotas('metadata.title')}
            </h1>
            <p className="text-muted-foreground mt-3 max-w-2xl text-sm leading-6">
              {page_quotas('metadata.description')}
            </p>
          </div>
        </div>
        <div className="mb-6 grid gap-3 md:grid-cols-3">
          <QuotaMetric
            icon={<Gauge className="size-4" />}
            label={page_quotas('metric_policies')}
            value={data.quotas.length}
          />
          <QuotaMetric
            icon={<TimerReset className="size-4" />}
            label={page_quotas('metric_usage')}
            value={totalUsage}
          />
          <QuotaMetric
            icon={<ShieldCheck className="size-4" />}
            label={page_quotas('metric_limit')}
            value={totalLimit}
          />
        </div>
        <QuotaChartGrid quotas={data.quotas} />
      </PageContent>
    </PageContainer>
  );
}

const QuotaMetric = ({
  icon,
  label,
  value,
}: {
  icon: ReactNode;
  label: string;
  value: number;
}) => {
  return (
    <Card className="border-border/70 gap-0 rounded-xl py-0">
      <CardContent className="flex items-center gap-3 p-4">
        <div className="bg-accent-soft text-accent-ink flex size-9 items-center justify-center rounded-lg">
          {icon}
        </div>
        <div>
          <div className="font-mono text-xl leading-none tabular-nums">
            {value}
          </div>
          <div className="text-muted-foreground mt-1 text-xs">{label}</div>
        </div>
      </CardContent>
    </Card>
  );
};
