import { BotSectionNav } from '@/components/evaluation/bot-section-nav';
import { EvaluationRunsPanel } from '@/components/evaluation/evaluation-runs-panel';
import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { listEvaluationRuns } from '@/features/evaluation/server-api';
import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';
import { getTranslations } from 'next-intl/server';
import { notFound } from 'next/navigation';

export default async function Page({
  params,
  searchParams,
}: {
  params: Promise<{ botId: string }>;
  searchParams?: Promise<{ datasetVersionId?: string }>;
}) {
  const { botId } = await params;
  const resolvedSearchParams = searchParams ? await searchParams : undefined;
  const serverApi = await getServerApi();
  const pageBot = await getTranslations('page_bot');
  const pageBotEvaluation = await getTranslations('page_bot_evaluation');

  let bot;

  try {
    const botRes = await serverApi.defaultApi.botsBotIdGet({
      botId,
    });
    bot = toJson(botRes.data);
  } catch {
    notFound();
  }

  const runs = await listEvaluationRuns(botId);

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          {
            title: pageBot('metadata.title'),
          },
          {
            title: bot.title || botId,
            href: `/workspace/bots/${botId}/chats`,
          },
          {
            title: pageBotEvaluation('metadata.title'),
          },
        ]}
      />
      <PageContent className="pb-0">
        <BotSectionNav botId={botId} active="evaluation" />
      </PageContent>
      <PageContent>
        <EvaluationRunsPanel
          bot={bot}
          runs={runs.items}
          unavailable={runs.unavailable}
          error={runs.error}
          initialDatasetVersionId={resolvedSearchParams?.datasetVersionId}
        />
      </PageContent>
    </PageContainer>
  );
}
