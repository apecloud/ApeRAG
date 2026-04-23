import { TitleGenerateRequestLanguageEnum } from '@/api';
import messages from './src/i18n/en-US.json';
import pageBenchmarks from './src/i18n/en-US/page_benchmarks.json';
import pageBotEvaluation from './src/i18n/en-US/page_bot_evaluation.json';

const typedMessages = {
  ...messages,
  page_benchmarks: pageBenchmarks,
  page_bot_evaluation: pageBotEvaluation,
};

declare module 'next-intl' {
  interface AppConfig {
    Messages: typeof typedMessages;
    Locale: TitleGenerateRequestLanguageEnum;
  }
}
