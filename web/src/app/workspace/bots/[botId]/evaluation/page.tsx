import { BotSectionNav } from '@/components/evaluation/bot-section-nav';
import { EvaluationRunsPanel } from '@/components/evaluation/evaluation-runs-panel';
import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { getBot } from '@/features/bot/server-api';
import { listEvaluationRuns } from '@/features/evaluation/server-api';
import { toJson } from '@/lib/utils';
import { getTranslations } from 'next-intl/server';
import { notFound } from 'next/navigation';

export default async function Page({
  params,
}: {
  params: Promise<{ botId: string }>;
}) {
  const { botId } = await params;
  const pageBot = await getTranslations('page_bot');
  const pageBotEvaluation = await getTranslations('page_bot_evaluation');

  let bot;

  try {
    const botRes = await getBot(botId);
    if (!botRes) {
      notFound();
    }
    bot = toJson(botRes);
  } catch {
    notFound();
  }

  const runs = await listEvaluationRuns({ botId });

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
        />
      </PageContent>
    </PageContainer>
  );
}
