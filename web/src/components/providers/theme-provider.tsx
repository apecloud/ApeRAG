'use client';
import { ThemeProvider as NextThemeProvider } from 'next-themes';
import { usePathname } from 'next/navigation';
import * as React from 'react';

export function ThemeProvider({
  children,
  ...props
}: React.ComponentProps<typeof NextThemeProvider>) {
  const pathname = usePathname();
  return (
    <NextThemeProvider
      {...props}
      forcedTheme={pathname === '/' ? 'dark' : undefined}
    >
      {children}
    </NextThemeProvider>
  );
}
