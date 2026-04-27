'use client';

import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Check,
  ChevronsUpDown,
  CircleQuestionMark,
  Globe,
  Moon,
  ShieldUser,
  Sun,
  User,
} from 'lucide-react';
import { useTheme } from 'next-themes';

import Link from 'next/link';

import { LogOut } from 'lucide-react';

import { useAppContext } from '@/components/providers/app-provider';
import { cn } from '@/lib/utils';

import { useIsMobile } from '@/hooks/use-mobile';
import { setLocale } from '@/services/cookies';
import { useLocale, useTranslations } from 'next-intl';
import { FaGithub } from 'react-icons/fa6';
import { NavigationMenu, NavigationMenuList } from './ui/navigation-menu';
import { UserAvatar, UserAvatarProfile } from './user-avatar';

/**
 * Full INEO Sage logo: orbital symbol + "INEO SAGE" wordmark
 * Adapts to current theme via fill-accent-foreground
 */
export const AppLogo = () => {
  return (
    <Link href="/" className="flex h-10 items-center">
      <svg
        viewBox="0 0 360 100"
        className="h-10 fill-accent-foreground"
        xmlns="http://www.w3.org/2000/svg"
        aria-label="INEO Sage"
      >
        {/* Orbital rings - two ellipses crossing diagonally */}
        <g transform="translate(50, 50)">
          <ellipse
            cx="0"
            cy="0"
            rx="38"
            ry="14"
            fill="none"
            stroke="currentColor"
            strokeWidth="3.5"
            transform="rotate(35)"
          />
          <ellipse
            cx="0"
            cy="0"
            rx="38"
            ry="14"
            fill="none"
            stroke="currentColor"
            strokeWidth="3.5"
            transform="rotate(-35)"
          />
          {/* Central figure: head + body */}
          <circle cx="0" cy="-18" r="7" />
          <path d="M -10,-8 Q 0,-12 10,-8 L 8,18 L -8,18 Z" />
        </g>

        {/* "INEO" wordmark - bold */}
        <text
          x="110"
          y="46"
          fontFamily="system-ui, -apple-system, 'Segoe UI', sans-serif"
          fontSize="32"
          fontWeight="700"
          letterSpacing="2"
        >
          INEO
        </text>

        {/* "SAGE" wordmark - lighter weight */}
        <text
          x="110"
          y="78"
          fontFamily="system-ui, -apple-system, 'Segoe UI', sans-serif"
          fontSize="28"
          fontWeight="300"
          letterSpacing="3"
        >
          SAGE
        </text>
      </svg>
    </Link>
  );
};

/**
 * Compact INEO Sage logo: just the orbital symbol (no text)
 * Used in mobile/collapsed sidebar contexts
 */
export const AppShortLogo = () => {
  return (
    <Link href="/" className="flex size-10 items-center">
      <svg
        viewBox="0 0 100 100"
        version="1.1"
        className="size-10 fill-accent-foreground"
        xmlns="http://www.w3.org/2000/svg"
        aria-label="INEO Sage"
      >
        <g transform="translate(50, 50)">
          <ellipse
            cx="0"
            cy="0"
            rx="38"
            ry="14"
            fill="none"
            stroke="currentColor"
            strokeWidth="3.5"
            transform="rotate(35)"
          />
          <ellipse
            cx="0"
            cy="0"
            rx="38"
            ry="14"
            fill="none"
            stroke="currentColor"
            strokeWidth="3.5"
            transform="rotate(-35)"
          />
          <circle cx="0" cy="-18" r="7" />
          <path d="M -10,-8 Q 0,-12 10,-8 L 8,18 L -8,18 Z" />
        </g>
      </svg>
    </Link>
  );
};

export const AppUserDropdownMenu = () => {
  const { user, signIn, signOut } = useAppContext();
  const username = user?.username || user?.email?.split('@')[0];
  const isMobile = useIsMobile();
  const locale = useLocale();
  const page_auth = useTranslations('page_auth');

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          className="data-[state=open]:bg-accent h-auto has-[>svg]:px-2"
        >
          <UserAvatar user={user} />
          <div className="grid flex-1 text-left text-sm leading-tight">
            <span className="max-w-30 truncate font-medium">{username}</span>
          </div>
          <ChevronsUpDown className="ml-auto size-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        className="w-(--radix-dropdown-menu-trigger-width) min-w-56 rounded-lg"
        align="end"
        side="bottom"
        sideOffset={isMobile ? 4 : 12}
      >
        {user && (
          <>
            <DropdownMenuLabel className="font-normal">
              <UserAvatarProfile user={user} />
            </DropdownMenuLabel>
            <DropdownMenuSeparator />
          </>
        )}

        <DropdownMenuGroup>
          <DropdownMenuItem onClick={() => setLocale('en-US')}>
            <Check
              data-active={locale === 'en-US'}
              className="opacity-0 data-[active=true]:opacity-100"
            />
            English
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => setLocale('pl-PL')}>
            <Check
              data-active={locale === 'pl-PL'}
              className="opacity-0 data-[active=true]:opacity-100"
            />
            Polski
          </DropdownMenuItem>
        </DropdownMenuGroup>

        <DropdownMenuSeparator />

        {user && (
          <>
            <DropdownMenuGroup>
              {user.role === 'admin' && (
                <DropdownMenuItem asChild>
                  <Link href="/admin">
                    <ShieldUser />
                    {page_auth('administrator')}
                  </Link>
                </DropdownMenuItem>
              )}
              <DropdownMenuItem disabled>
                <User />
                {page_auth('account')}
              </DropdownMenuItem>
              <DropdownMenuSeparator />
            </DropdownMenuGroup>
          </>
        )}

        {user ? (
          <DropdownMenuItem onClick={signOut}>
            <LogOut />
            {page_auth('signout')}
          </DropdownMenuItem>
        ) : (
          <DropdownMenuItem onClick={() => signIn()}>
            <LogOut />
            {page_auth('signin')}
          </DropdownMenuItem>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

export const AppThemeDropdownMenu = () => {
  const { setTheme } = useTheme();
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="data-[state=open]:bg-accent"
        >
          <Sun className="h-[1.2rem] w-[1.2rem] scale-100 rotate-0 transition-all dark:scale-0 dark:-rotate-90" />
          <Moon className="absolute h-[1.2rem] w-[1.2rem] scale-0 rotate-90 transition-all dark:scale-100 dark:rotate-0" />
          <span className="sr-only">Toggle theme</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent side="bottom" align="end">
        <DropdownMenuItem onClick={() => setTheme('light')}>
          Light
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => setTheme('dark')}>
          Dark
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => setTheme('system')}>
          System
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

export const AppLocaleDropdownMenu = () => {
  const locale = useLocale();
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="data-[state=open]:bg-accent"
        >
          <Globe />
          <span className="sr-only">Toggle locale</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent side="bottom" align="end">
        <DropdownMenuItem onClick={() => setLocale('en-US')}>
          <Check
            data-active={locale === 'en-US'}
            className="opacity-0 data-[active=true]:opacity-100"
          />
          English
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => setLocale('pl-PL')}>
          <Check
            data-active={locale === 'pl-PL'}
            className="opacity-0 data-[active=true]:opacity-100"
          />
          Polski
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

export const AppDocs = () => (
  <Button variant="ghost" size="icon" asChild>
    <Link href="/docs">
      <CircleQuestionMark />
      <span className="sr-only">Documents</span>
    </Link>
  </Button>
);

export const AppGithub = () => (
  <Button variant="ghost" size="icon" asChild>
    <Link target="_blank" href="https://github.com/mirekwilmer/INEOSage">
      <FaGithub />
      <span className="sr-only">Github</span>
    </Link>
  </Button>
);

export const AppTopbar = ({ className }: React.ComponentProps<'div'>) => {
  return (
    <>
      <header
        className={cn(
          'fixed z-40 flex h-16 w-full shrink-0 items-center justify-between gap-2 px-4 backdrop-blur-lg transition-[width,height] ease-linear',
          className,
        )}
      >
        <div className="flex items-center gap-8">
          <AppLogo />
          <NavigationMenu>
            <NavigationMenuList>
              {/* <NavigationMenuItem>
                <NavigationMenuLink asChild className="hover:bg-accent/30 px-4">
                  <Link href="/marketplace">Marketplace</Link>
                </NavigationMenuLink>
              </NavigationMenuItem> */}
            </NavigationMenuList>
          </NavigationMenu>
        </div>
        <div className="flex flex-row items-center gap-2">
          <AppGithub />
          <AppDocs />
          <AppThemeDropdownMenu />
          <AppUserDropdownMenu />
        </div>
      </header>
    </>
  );
};