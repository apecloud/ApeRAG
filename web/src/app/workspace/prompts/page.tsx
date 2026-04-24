import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { Card, CardContent } from '@/components/ui/card';
import { getUserPrompts } from '@/features/prompt/server-api';
import { toJson } from '@/lib/utils';
import { FileText, Settings2, Sparkles } from 'lucide-react';
import { getTranslations } from 'next-intl/server';
import type { ReactNode } from 'react';
import { PromptSettings } from './prompt-settings';

const PROMPT_TEMPLATE_COUNT = 2;

export default async function Page() {
  const data = await getUserPrompts();
  const page_prompts = await getTranslations('page_prompts');
  const promptEntries = Object.values(data ?? {});
  const customizedCount = promptEntries.filter(
    (detail) => detail?.customized,
  ).length;

  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: page_prompts('metadata.title') }]} />
      <PageContent className="max-w-7xl px-5 py-8 md:px-8 md:py-10">
        <div className="mb-8 flex flex-col gap-5 lg:flex-row lg:items-end">
          <div className="min-w-0 flex-1">
            <div className="text-muted-foreground font-mono text-[11px] tracking-[0.12em] uppercase">
              {page_prompts('metadata.label')}
            </div>
            <h1 className="font-serif mt-2 text-4xl leading-none font-normal tracking-normal md:text-[44px]">
              {page_prompts('metadata.title')}
            </h1>
            <p className="text-muted-foreground mt-3 max-w-2xl text-sm leading-6">
              {page_prompts('metadata.description')}
            </p>
          </div>
        </div>

        <div className="mb-6 grid gap-3 md:grid-cols-3">
          <PromptMetric
            icon={<FileText className="size-4" />}
            label={page_prompts('metric.templates')}
            value={PROMPT_TEMPLATE_COUNT}
          />
          <PromptMetric
            icon={<Sparkles className="size-4" />}
            label={page_prompts('metric.customized')}
            value={customizedCount}
          />
          <PromptMetric
            icon={<Settings2 className="size-4" />}
            label={page_prompts('metric.defaults')}
            value={PROMPT_TEMPLATE_COUNT - customizedCount}
          />
        </div>

        <PromptSettings data={toJson(data)} />
      </PageContent>
    </PageContainer>
  );
}

const PromptMetric = ({
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
