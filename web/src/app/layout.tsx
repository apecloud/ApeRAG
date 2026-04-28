import { Toaster } from '@/components/ui/sonner';
import { getCurrentUser } from '@/features/auth/server-api';
import type { Metadata } from 'next';
import { NextIntlClientProvider } from 'next-intl';
import { Fraunces, JetBrains_Mono, Manrope } from 'next/font/google';
import NextTopLoader from 'nextjs-toploader';

import { AppProvider } from '@/components/providers/app-provider';
import { ThemeProvider } from '@/components/providers/theme-provider';
import { getLocale } from '@/services/cookies';
import 'highlight.js/styles/github-dark.css';
import './globals.css';

import { getTranslations } from 'next-intl/server';

const fontSans = Manrope({
  variable: '--font-sans',
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  display: 'swap',
});

const fontSerif = Fraunces({
  variable: '--font-serif',
  subsets: ['latin'],
  axes: ['opsz'],
  display: 'swap',
});

const fontMono = JetBrains_Mono({
  variable: '--font-mono',
  subsets: ['latin'],
  weight: ['400', '500'],
  display: 'swap',
});

export async function generateMetadata(): Promise<Metadata> {
  const common_site = await getTranslations('common.site');
  return {
    applicationName: common_site('metadata.applicationName'),
    authors: {
      name: common_site('metadata.authors.name'),
      url: common_site('metadata.authors.url'),
    },
    title: {
      default: common_site('metadata.title'),
      template: `%s | ${common_site('metadata.title')}`,
    },
    description: common_site('metadata.description'),
    keywords: ['RAG', 'Graph Search', 'Vector Search', 'Full-Text Search'],
  };
}

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const user = (await getCurrentUser()) ?? undefined;
  const locale = await getLocale();

  return (
    <html lang={locale} suppressHydrationWarning>
      <body
        className={`${fontSans.variable} ${fontSerif.variable} ${fontMono.variable} font-sans antialiased`}
      >
        <NextTopLoader
          // color="color-mix(in oklab, var(--primary), transparent)"
          color="var(--primary)"
          showSpinner={false}
          crawl={false}
        />
        <NextIntlClientProvider>
          <ThemeProvider
            attribute="class"
            defaultTheme={process.env.NEXT_PUBLIC_DEFAULT_THEME || 'light'}
            enableSystem
            disableTransitionOnChange
          >
            <Toaster position="top-center" richColors />
            <AppProvider user={user}>{children}</AppProvider>
          </ThemeProvider>
        </NextIntlClientProvider>
      </body>
    </html>
  );
}
