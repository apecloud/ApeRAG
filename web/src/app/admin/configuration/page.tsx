import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { Card, CardContent } from '@/components/ui/card';
import {
  getSettings,
  getSystemDefaultQuotas,
} from '@/features/admin/server-api';
import { FileSearch, Gauge, SlidersHorizontal } from 'lucide-react';
import { Metadata } from 'next';
import { getTranslations } from 'next-intl/server';
import type { ReactNode } from 'react';
import { ParserSettings } from './parser-settings';
import { QuotaSettings } from './quota-settings';

export async function generateMetadata(): Promise<Metadata> {
  const admin_config = await getTranslations('admin_config');
  return {
    title: admin_config('metadata.title'),
    description: admin_config('metadata.description'),
  };
}

export default async function Page() {
  const admin_config = await getTranslations('admin_config');

  const [settings, systemDefaultQuotas] = await Promise.all([
    getSettings(),
    getSystemDefaultQuotas(),
  ]);
  const activeParserModes = [
    settings?.use_markitdown,
    settings?.use_mineru,
  ].filter(Boolean).length;
  const quotaPolicyCount = Object.keys(
    systemDefaultQuotas?.quotas ?? {},
  ).length;
  const configurationPanels = 3 + (systemDefaultQuotas?.quotas ? 1 : 0);

  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: admin_config('metadata.title') }]} />
      <PageContent className="max-w-7xl px-5 py-8 md:px-8 md:py-10">
        <div className="mb-8 flex flex-col gap-5 lg:flex-row lg:items-end">
          <div className="min-w-0 flex-1">
            <div className="text-muted-foreground font-mono text-[11px] tracking-[0.12em] uppercase">
              {admin_config('metadata.label')}
            </div>
            <h1 className="mt-2 font-serif text-4xl leading-none font-normal tracking-normal md:text-[44px]">
              {admin_config('metadata.title')}
            </h1>
            <p className="text-muted-foreground mt-3 max-w-2xl text-sm leading-6">
              {admin_config('metadata.description')}
            </p>
          </div>
        </div>

        <div className="mb-6 grid gap-3 md:grid-cols-3">
          <ConfigMetric
            icon={<FileSearch className="size-4" />}
            label={admin_config('metric_parser_modes')}
            value={activeParserModes}
          />
          <ConfigMetric
            icon={<Gauge className="size-4" />}
            label={admin_config('metric_quota_policies')}
            value={quotaPolicyCount}
          />
          <ConfigMetric
            icon={<SlidersHorizontal className="size-4" />}
            label={admin_config('metric_control_sections')}
            value={configurationPanels}
          />
        </div>

        <div className="flex flex-col gap-6">
          <ParserSettings data={settings ?? undefined} />
          {systemDefaultQuotas?.quotas ? (
            <QuotaSettings data={systemDefaultQuotas.quotas} />
          ) : null}
        </div>
      </PageContent>
    </PageContainer>
  );
}

const ConfigMetric = ({
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
