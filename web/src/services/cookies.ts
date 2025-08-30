'use server';

import { cookies } from 'next/headers';

const localeCookieName = 'locale';
const locales = ['en-US', 'zh-CN'] as const;

type LocaleEnum = (typeof locales)[number];

const defaultLocale: LocaleEnum = (process.env.NEXT_PUBLIC_DEFAULT_LOCALE ||
  'en-US') as LocaleEnum;

/**
 * get locale
 * In this example the locale is read from a cookie. You could alternatively
 * also read it from a database, backend service, or any other source.
 */
export async function getLocale(): Promise<LocaleEnum> {
  const cookieLocale = ((await cookies()).get(localeCookieName)?.value ||
    defaultLocale) as LocaleEnum;

  return locales.includes(cookieLocale) ? cookieLocale : defaultLocale;
}

/**
 * set locale
 */
export async function setLocale(locale: LocaleEnum) {
  (await cookies()).set(localeCookieName, locale);
}

/**
 * get cookie by name
 */
export async function getCookie(name: string): Promise<string | undefined> {
  return (await cookies()).get(name)?.value;
}
