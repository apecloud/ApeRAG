import { Toaster } from '@/components/ui/sonner';
import { getCurrentUser } from '@/features/auth/server-api';
import type { Metadata } from 'next';
import { NextIntlClientProvider } from 'next-intl';
import { Geist, Geist_Mono } from 'next/font/google';
import NextTopLoader from 'nextjs-toploader';

import { AppProvider } from '@/components/providers/app-provider';
import { ThemeProvider } from '@/components/providers/theme-provider';
import 'highlight.js/styles/github-dark.css';
import './globals.css';

import { getLocale, getTranslations } from 'next-intl/server';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
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

  // Get current locale and convert to short HTML lang code (e.g., 'en-US' -> 'en')
  const locale = await getLocale();
  const htmlLang = locale.split('-')[0] || 'en';

  return (
    <html lang={htmlLang} suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <NextTopLoader
          color="var(--primary)"
          showSpinner={false}
          crawl={false}
        />
        <NextIntlClientProvider>
          <ThemeProvider
            attribute="class"
            defaultTheme={process.env.NEXT_PUBLIC_DEFAULT_THEME || 'system'}
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