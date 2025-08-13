/**
 * Locale
 */
export const localeCookieName = 'locale';
export const locales = ['en-US', 'zh-CN'] as const;
export type LocaleEnum = (typeof locales)[number];
export const defaultLocale: LocaleEnum = 'en-US';
