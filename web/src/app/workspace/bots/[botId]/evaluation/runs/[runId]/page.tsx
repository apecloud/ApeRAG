import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { BotSectionNav } from '@/components/evaluation/bot-section-nav';
import { EvaluationRunDetail } from '@/components/evaluation/evaluation-run-detail';
import {
  getEvaluationRunDetail,
  listEvaluationRunItems,
} from '@/components/evaluation/server';
import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';
import { getTranslations } from 'next-intl/server';
import { notFound } from 'next/navigation';

export default async function Page({
  params,
}: {
  params: Promise<{ botId: string; runId: string }>;
}) {
  const { botId, runId } = await params;
  const [runDetail, runItems] = await Promise.all([
    getEvaluationRunDetail(runId),
    listEvaluationRunItems(runId),
  ]);
  const resolvedBotId = runDetail.payload?.run?.bot_id || botId;

  if (runDetail.payload?.run?.bot_id && runDetail.payload.run.bot_id !== botId) {
    notFound();
  }

  const serverApi = await getServerApi();
  const pageBot = await getTranslations('page_bot');
  const pageBotEvaluation = await getTranslations('page_bot_evaluation');

  let bot;

  try {
    const botRes = await serverApi.defaultApi.botsBotIdGet({
      botId: resolvedBotId,
    });
    bot = toJson(botRes.data);
  } catch {
    notFound();
  }

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          {
            title: pageBot('metadata.title'),
          },
          {
            title: bot.title || resolvedBotId,
            href: `/workspace/bots/${resolvedBotId}/evaluation`,
          },
          {
            title: pageBotEvaluation('metadata.title'),
            href: `/workspace/bots/${resolvedBotId}/evaluation`,
          },
          {
            title: runId,
          },
        ]}
      />
      <PageContent className="pb-0">
        <BotSectionNav botId={resolvedBotId} active="evaluation" />
      </PageContent>
      <PageContent>
        <EvaluationRunDetail
          botId={resolvedBotId}
          detail={runDetail.payload}
          items={runItems.items}
          unavailable={runDetail.unavailable || runItems.unavailable}
          error={runDetail.error || runItems.error}
        />
      </PageContent>
    </PageContainer>
  );
}
