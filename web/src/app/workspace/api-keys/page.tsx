import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { Card, CardContent } from '@/components/ui/card';

import { listApiKeys } from '@/features/api-key/server-api';
import { toJson } from '@/lib/utils';
import { KeyRound, ShieldCheck, TimerReset } from 'lucide-react';
import { getTranslations } from 'next-intl/server';
import type { ReactNode } from 'react';
import { ApiKeyTable } from './api-key-table';

export default async function Page() {
  const data = await listApiKeys();
  const page_api_keys = await getTranslations('page_api_keys');
  const usedKeys = data.filter((item) => item.last_used_at).length;

  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: page_api_keys('metadata.title') }]} />
      <PageContent className="max-w-7xl px-5 py-8 md:px-8 md:py-10">
        <div className="mb-8 flex flex-col gap-5 lg:flex-row lg:items-end">
          <div className="min-w-0 flex-1">
            <div className="text-muted-foreground font-mono text-[11px] tracking-[0.12em] uppercase">
              {page_api_keys('metadata.label')}
            </div>
            <h1 className="mt-2 font-serif text-4xl leading-none font-normal tracking-normal md:text-[44px]">
              {page_api_keys('metadata.title')}
            </h1>
            <p className="text-muted-foreground mt-3 max-w-2xl text-sm leading-6">
              {page_api_keys('metadata.description')}
            </p>
          </div>
        </div>
        <div className="mb-6 grid gap-3 md:grid-cols-3">
          <GovernanceMetric
            icon={<KeyRound className="size-4" />}
            label={page_api_keys('metric_total')}
            value={data.length}
          />
          <GovernanceMetric
            icon={<TimerReset className="size-4" />}
            label={page_api_keys('metric_used')}
            value={usedKeys}
          />
          <GovernanceMetric
            icon={<ShieldCheck className="size-4" />}
            label={page_api_keys('metric_masked')}
            value={data.length}
          />
        </div>
        <ApiKeyTable data={toJson(data)} />
      </PageContent>
    </PageContainer>
  );
}

const GovernanceMetric = ({
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
