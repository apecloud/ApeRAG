import { Toaster } from '@/components/ui/sonner';
import { getServerApi } from '@/lib/api/server';
import type { Metadata } from 'next';
import { NextIntlClientProvider } from 'next-intl';
import { Geist, Geist_Mono } from 'next/font/google';
import NextTopLoader from 'nextjs-toploader';
<<<<<<< HEAD

import { AppProvider } from '@/components/app-provider';
import { ThemeProvider } from '@/components/theme-provider';

import 'highlight.js/styles/github-dark.css';
=======
import { GlobalProvider } from './global-provider';
>>>>>>> 87ec7d8e (fix: add topbar loader)
import './globals.css';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: 'ApeRAG',
  description:
    'Production-Ready RAG Platform with Graph, Vector & Full-Text Search',
};

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  let user;
  const apiServer = await getServerApi();
  try {
    const res = await apiServer.defaultApi.userGet();
    user = res.data;
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
  } catch (err) {}

  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <NextTopLoader
          color="color-mix(in oklab, var(--primary) 30%, transparent)"
          showSpinner={false}
          crawl={false}
        />
        <NextIntlClientProvider>
          <ThemeProvider
            attribute="class"
            defaultTheme="system"
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
