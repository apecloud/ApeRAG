import type { TitleLanguage } from '@/features/bot/types';
import messages from './src/i18n/en-US.json';
import pageBotEvaluation from './src/i18n/en-US/page_bot_evaluation.json';
import pageCollectionEvaluations from './src/i18n/en-US/page_collection_evaluations.json';

const typedMessages = {
  ...messages,
  page_bot_evaluation: pageBotEvaluation,
  page_collection_evaluations: pageCollectionEvaluations,
};

declare module 'next-intl' {
  interface AppConfig {
    Messages: typeof typedMessages;
    Locale: TitleLanguage;
  }
}
