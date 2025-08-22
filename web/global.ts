import { TitleGenerateRequestLanguageEnum } from '@/api';
import messages from './src/i18n/en-US.json';

declare module 'next-intl' {
  interface AppConfig {
    Messages: typeof messages;
    Locale: TitleGenerateRequestLanguageEnum;
  }
}
