import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { Card, CardContent } from '@/components/ui/card';
import { getProviderCatalog } from '@/features/providers/server-api';
import { toJson } from '@/lib/utils';
import { Boxes, BrainCircuit, Globe2, KeyRound } from 'lucide-react';
import { getTranslations } from 'next-intl/server';
import type { ReactNode } from 'react';
import { ProviderTable } from './provider-table';

export default async function Page() {
  const page_models = await getTranslations('page_models');
  const providerCatalog = await getProviderCatalog();
  const providers = providerCatalog.providers;
  const models = providerCatalog.models;

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[{ title: page_models('metadata.provider_title') }]}
      />
      <PageContent className="max-w-7xl px-5 py-8 md:px-8 md:py-10">
        <div className="mb-8 flex flex-col gap-5 lg:flex-row lg:items-end">
          <div className="min-w-0 flex-1">
            <div className="text-muted-foreground font-mono text-[11px] tracking-[0.12em] uppercase">
              {page_models('metadata.provider_label')}
            </div>
            <h1 className="font-serif mt-2 text-4xl leading-none font-normal tracking-normal md:text-[44px]">
              {page_models('metadata.provider_title')}
            </h1>
            <p className="text-muted-foreground mt-3 max-w-2xl text-sm leading-6">
              {page_models('metadata.provider_description')}
            </p>
          </div>
        </div>

        <div className="mb-6 grid gap-3 md:grid-cols-4">
          <ProviderMetric
            icon={<Boxes className="size-4" />}
            label={page_models('provider.metric_providers')}
            value={providers.length}
          />
          <ProviderMetric
            icon={<KeyRound className="size-4" />}
            label={page_models('provider.metric_enabled')}
            value={providers.filter((provider) => provider.api_key).length}
          />
          <ProviderMetric
            icon={<Globe2 className="size-4" />}
            label={page_models('provider.metric_public')}
            value={
              providers.filter((provider) => provider.user_id === 'public')
                .length
            }
          />
          <ProviderMetric
            icon={<BrainCircuit className="size-4" />}
            label={page_models('provider.metric_models')}
            value={models.length}
          />
        </div>

        <ProviderTable
          data={toJson(providers)}
          models={toJson(models)}
          urlPrefix="/workspace"
        />
      </PageContent>
    </PageContainer>
  );
}

const ProviderMetric = ({
  icon,
  label,
  value,
}: {
  icon: ReactNode;
  label: string;
  value: number;
}) => {
  return (
    <Card className="gap-0 rounded-xl border-border/70 py-0">
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
