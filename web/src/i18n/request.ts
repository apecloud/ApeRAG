import { getLocale } from '@/services/cookies';
import { getRequestConfig } from 'next-intl/server';

/**
 * Map locales to their appropriate timezones.
 * Falls back to UTC for unknown locales.
 */
const LOCALE_TIMEZONE_MAP: Record<string, string> = {
  'en-US': 'UTC',
  'pl-PL': 'Europe/Warsaw',
};

export default getRequestConfig(async () => {
  const locale = await getLocale();

  // Determine timezone based on locale
  const timeZone = LOCALE_TIMEZONE_MAP[locale] || 'UTC';

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