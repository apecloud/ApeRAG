'use server';

import { cookies } from 'next/headers';

const localeCookieName = 'locale';

/**
 * Supported application locales.
 * To add a new language:
 *   1. Add the locale code here (e.g., 'de-DE')
 *   2. Create src/i18n/{locale}.json with translations
 *   3. Create src/i18n/{locale}/ namespace folder
 *   4. Update LOCALE_TIMEZONE_MAP in src/i18n/request.ts
 *   5. Add a menu item in src/components/app-topbar.tsx
 */
const locales = ['en-US', 'pl-PL'] as const;

export type LocaleEnum = (typeof locales)[number];

const defaultLocale: LocaleEnum = (process.env.NEXT_PUBLIC_DEFAULT_LOCALE ||
  'en-US') as LocaleEnum;

/**
 * Type guard to safely check if a string is a valid locale.
 */
function isValidLocale(value: string): value is LocaleEnum {
  return (locales as readonly string[]).includes(value);
}

/**
 * Get locale.
 * In this example the locale is read from a cookie. You could alternatively
 * also read it from a database, backend service, or any other source.
 */
export async function getLocale(): Promise<LocaleEnum> {
  const cookieLocale = (await cookies()).get(localeCookieName)?.value;

  if (cookieLocale && isValidLocale(cookieLocale)) {
    return cookieLocale;
  }

  return defaultLocale;
}

/**
 * Set locale.
 */
export async function setLocale(locale: LocaleEnum) {
  (await cookies()).set(localeCookieName, locale, {
    // Persist for 1 year
    maxAge: 60 * 60 * 24 * 365,
    // Available across the whole app
    path: '/',
    // Allow client-side reading if needed
    httpOnly: false,
    sameSite: 'lax',
  });
}

/**
 * Get cookie by name.
 */
export async function getCookie(name: string): Promise<string | undefined> {
  return (await cookies()).get(name)?.value;
}