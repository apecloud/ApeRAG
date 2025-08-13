import { getLocale } from '@/services/cookies';
import { getRequestConfig } from 'next-intl/server';

export default getRequestConfig(async () => {
  const locale = await getLocale();

  /**
   * user timezone
   */
  const timeZone = 'Asia/Shanghai';

  return {
    locale,
    messages: (await import(`./${locale}.json`)).default,
    formats: {
      dateTime: {
        full: {
          timeStyle: 'full',
          dateStyle: 'full',
        },
        long: {
          timeStyle: 'long',
          dateStyle: 'long',
        },
        medium: {
          timeStyle: 'medium',
          dateStyle: 'medium',
        },
        short: {
          timeStyle: 'short',
          dateStyle: 'short',
        },
      },
    },
    timeZone,
  };
});
