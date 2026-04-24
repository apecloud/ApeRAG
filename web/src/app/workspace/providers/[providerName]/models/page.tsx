import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { Card, CardContent } from '@/components/ui/card';
import {
  getProvider,
  getProviderModels,
} from '@/features/providers/server-api';

import { Boxes, BrainCircuit, GitBranch, Layers3 } from 'lucide-react';
import { getTranslations } from 'next-intl/server';
import type { ReactNode } from 'react';
import { ModelTable } from './model-table';

export default async function Page({
  params,
}: {
  params: Promise<{ providerName: string }>;
}) {
  const { providerName } = await params;
  const page_models = await getTranslations('page_models');

  const [models, provider] = await Promise.all([
    getProviderModels(providerName),
    getProvider(providerName),
  ]);
  const completionCount = models.filter(
    (model) => model.api === 'completion',
  ).length;
  const embeddingCount = models.filter(
    (model) => model.api === 'embedding',
  ).length;
  const rerankCount = models.filter((model) => model.api === 'rerank').length;

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          {
            title: page_models('metadata.provider_title'),
            href: '/workspace/providers',
          },
          { title: provider?.label ?? providerName },
          { title: page_models('metadata.model_title') },
        ]}
      />
      <PageContent className="max-w-7xl px-5 py-8 md:px-8 md:py-10">
        <div className="mb-8 flex flex-col gap-5 lg:flex-row lg:items-end">
          <div className="min-w-0 flex-1">
            <div className="text-muted-foreground font-mono text-[11px] tracking-[0.12em] uppercase">
              {page_models('metadata.model_label')}
            </div>
            <h1 className="font-serif mt-2 text-4xl leading-none font-normal tracking-normal md:text-[44px]">
              {provider?.label ?? providerName}
            </h1>
            <p className="text-muted-foreground mt-3 max-w-2xl text-sm leading-6">
              {page_models('metadata.model_description')}
            </p>
          </div>
        </div>

        <div className="mb-6 grid gap-3 md:grid-cols-4">
          <ModelMetric
            icon={<Boxes className="size-4" />}
            label={page_models('model.metric_total')}
            value={models.length}
          />
          <ModelMetric
            icon={<BrainCircuit className="size-4" />}
            label={page_models('model.metric_completion')}
            value={completionCount}
          />
          <ModelMetric
            icon={<Layers3 className="size-4" />}
            label={page_models('model.metric_embedding')}
            value={embeddingCount}
          />
          <ModelMetric
            icon={<GitBranch className="size-4" />}
            label={page_models('model.metric_rerank')}
            value={rerankCount}
          />
        </div>

        <ModelTable
          provider={provider}
          data={models}
          pathnamePrefix="/workspace"
        />
      </PageContent>
    </PageContainer>
  );
}

const ModelMetric = ({
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
