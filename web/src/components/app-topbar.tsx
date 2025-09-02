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

export const AppLogo = () => {
  return (
    <Link href="/" className="flex h-8 w-32 items-center">
      <svg viewBox="0 0 3840 1024" className="fill-accent-foreground">
        <path d="M2998.7584 706.944c0.3072 2.6368-1.7664 5.9648-6.144 9.9584-4.4288 3.968-8.3712 5.9648-11.9296 5.9648h-37.1456l47.2832-185.856c0.5376-3.0976 2.7904-6.528 6.784-10.2912 3.968-3.7632 7.7312-5.632 11.264-5.632h36.5056l-46.6176 185.856z m-55.7312-219.6736c-7.9872 0-15.1552 1.2288-21.504 3.6608a81.152 81.152 0 0 0-19.3024 10.9568 73.472 73.472 0 0 0-28.1088 43.776l-48.5632 210.9952h220.9536c7.9616 0 15.1296-1.2032 21.4528-3.6352 6.3744-2.432 12.8-6.0672 19.328-10.9568a73.3952 73.3952 0 0 0 28.1088-43.776l48.5632-211.0208h-220.928zM3425.536 487.2704c-37.6064 0-59.7504 19.456-66.4576 58.4192l-40.3456 177.1776c-11.9552 0-21.7088-2.8672-29.2352-8.6272a33.28 33.28 0 0 1-13.3888-21.888 28.4416 28.4416 0 0 1 0.3072-10.624l43.904-191.7952h-107.3664l-46.08 196.4032c0.0768 0.4352-0.3584 3.8656-1.3568 10.2912-0.9472 6.4-0.9472 12.928 0.1024 19.584 1.8944 11.9296 6.6304 21.6576 14.208 29.184 7.6032 7.5264 19.7888 11.264 36.6336 11.264h201.7024l63.3088-269.3888h-55.936zM2201.7536 521.088h36.2496c5.2992 0 9.3952 1.024 11.008 7.6544a22.528 22.528 0 0 1-0.512 8.2688l-14.3872 61.696H2183.68l18.0736-77.6192z m159.1552-2.6368a41.0624 41.0624 0 0 0-9.1904-17.2544c-4.608-5.2736-9.1392-8.96-13.6192-10.9312-4.4544-1.9968-10.88-2.9952-19.2768-2.9952H2096.128l-49.7664 210.9952c-2.5344 11.52-2.9696 20.5824-1.3312 27.2128 1.5104 6.1952 4.5568 11.9552 9.1648 17.2544 4.608 5.3248 9.1648 8.96 13.6192 10.9568 4.4544 1.9968 10.88 2.9952 19.3024 2.9952h217.7024l8.4224-33.8176h-144c-5.76 0-9.4464-3.328-11.0336-9.984a10.3936 10.3936 0 0 1 0.512-5.9648l15.7952-74.3936h41.9328c0.3328 0 0.6656 0.0768 1.024 0.0768h119.424l22.656-86.9376c2.5088-11.4944 2.944-20.5568 1.3312-27.2128zM1880.9344 706.944c-0.4096 3.072-2.5344 6.528-6.3488 10.2912-3.84 3.7632-7.5008 5.632-11.008 5.632h-36.5312l43.52-186.496c-0.4352-2.6368 1.4592-5.9392 5.7088-9.9328 4.224-3.9936 8.1152-5.9904 11.648-5.9904h37.1456l-44.1344 186.496z m-59.9808-219.6736c-7.9616 0-15.0528 1.2288-21.2992 3.6608-6.2464 2.432-12.544 6.0928-18.8416 10.9568a69.5296 69.5296 0 0 0-26.24 43.776l-75.4432 318.6432h61.9008c7.9872 0 15.104-1.2288 21.3248-3.6608 6.272-2.432 12.544-6.0928 18.8416-10.9312a69.5808 69.5808 0 0 0 26.24-43.8016l11.4944-49.2544h111.7696c7.9872 0 15.0784-1.2032 21.3248-3.6352 6.2464-2.432 12.5184-6.0672 18.8416-10.9568a69.5552 69.5552 0 0 0 26.24-43.776l49.792-211.0208H1820.928zM1612.0832 471.9616c0.3584 2.6624-1.6384 5.9904-5.9904 9.984-4.3264 3.968-8.2432 5.9392-11.776 5.9392h-56.32l19.8144-93.6704c0.512-3.072 2.6624-6.5024 6.6048-10.2656 3.9168-3.7632 7.6288-5.632 11.1872-5.632h57.6l-21.12 93.6448z m-102.7328-127.4624c-7.936 0-15.104 1.2288-21.4016 3.6608a79.616 79.616 0 0 0-19.0976 10.9312 71.6544 71.6544 0 0 0-27.3152 43.8016l-67.328 296.704-0.128 0.5632-6.0416 25.984a6.4768 6.4768 0 0 1-0.8448 3.6864l-6.2464 26.8544h60.2624c7.9872 0 15.1296-1.2288 21.4272-3.6608 6.2976-2.4064 12.672-6.0672 19.0976-10.9312a71.68 71.68 0 0 0 27.3152-43.8016l41.472-176.6144h69.5808l-39.0144 166.656c-7.68 34.4576 19.8656 68.352 55.552 68.352h38.7072l94.7456-412.16h-240.7424zM3640.9856 706.944c0.3328 2.6624-1.7152 5.9648-6.1184 9.9584-4.3776 3.968-8.3456 5.9648-11.904 5.9648h-37.1456l42.6752-185.2416c0.512-3.0976 2.7648-6.528 6.7584-10.2912 3.968-3.7632 7.7056-5.632 11.264-5.632h37.8112l-43.3408 185.2416zM3840 344.4992h-59.52a59.648 59.648 0 0 0-21.4784 3.6608 80.8448 80.8448 0 0 0-19.2512 10.9312 73.3952 73.3952 0 0 0-28.032 43.776l-18.5344 84.4032h-107.8784c-7.9616 0-15.104 1.2288-21.4784 3.6608a80.9472 80.9472 0 0 0-19.2768 10.9568 73.1392 73.1392 0 0 0-27.9808 43.776l-48.7168 211.0208h220.9792c7.936 0 15.104-1.2288 21.4528-3.6608 6.3488-2.432 12.7744-6.0672 19.2768-10.9312a73.0368 73.0368 0 0 0 27.9808-43.8016L3840 344.4992zM2636.4416 722.8672h-45.2352c-25.6512 0-50.7904-14.1568-75.392-42.4704a50.7392 50.7392 0 0 1-12.6976-28.544c-1.3824-11.4944 0.3072-27.1872 4.992-47.104l48.384-208.7424c0.512-3.072 2.7392-6.5024 6.7328-10.24 3.968-3.7888 7.7312-5.6576 11.264-5.6576h141.9264l8.2944-35.6096H2508.032c-7.9616 0-15.104 1.2288-21.4528 3.6608a81.5872 81.5872 0 0 0-19.2768 10.9568 73.0624 73.0624 0 0 0-27.9808 43.776l-44.4416 194.5344c-3.0464 15.488-3.8912 28.7488-2.56 39.808 1.4848 12.3904 5.12 24.5504 10.9824 36.5056 5.8368 11.9296 13.5936 23.9872 23.2192 36.1472 9.6256 12.1856 23.9616 23.04 43.008 32.512 19.0464 9.5232 35.8912 14.2592 50.4832 14.2592h108.5184l7.9104-33.792zM2776.1664 756.6848h-104.9088l96.1536-412.1856h104.9344l-96.1792 412.16zM1066.112 722.6624c-15.0016-74.2912-63.744-133.5808-106.7264-193.6384-48.384-67.6608-88.96-140.672-139.5712-206.7712-45.8496-59.904-101.0944-119.7312-180.5568-127.5904-48.7424-4.8128-95.6672 9.8816-137.856 33.92-17.28 10.5472-99.4304 65.792-130.1504 175.232a71.04 71.04 0 0 0-18.1248 7.7568l-0.3328 0.1792c-0.128 0.0512-0.2048 0.128-0.3072 0.2048-6.1696 3.1744-12.0832 7.168-17.7152 11.8528-40.8832 29.6192-68.8128 69.76-87.7056 106.7264l-0.1024 0.0512s-12.544 22.5536-22.2976 52.608c-2.304 6.7328-4.1728 12.7232-5.7856 18.2784C139.9552 545.536 43.776 569.6 0 657.2544c23.1168 3.2 47.1552 3.84 69.632 10.0864 75.1104 20.8384 110.848 82.048 144.3584 145.1008 18.816 35.4816 35.456 72.6272 58.24 105.3952 66.5088 95.744 216.9088 118.4256 324.608 48.7424-69.376-13.9264-143.2576-59.648-182.1952-110.08-49.5616-64.1536-126.4896-190.1056-170.1376-233.6512-0.5376-0.5376-1.1264-0.9728-1.664-1.5104 13.568-25.0368 32.384-49.664 55.3472-54.0672 21.5808-1.92 41.984 4.608 58.7776 20.5312 17.2032 16.3072 34.816 38.528 52.0704 68.608 83.2768 145.4848 186.7008 247.552 359.5776 206.4384-12.672-67.328-63.232-154.8544-63.2832-154.9312-36.6592-60.672-86.7072-140.3136-112.8448-172.5696-35.2256-43.4432-71.04-91.2128-121.5232-117.1712a162.8928 162.8928 0 0 0-44.6464-15.36c55.2704-45.6192 138.6752-81.408 213.7088 19.8912 61.5168 83.584 180.3008 256.0256 225.28 405.8112 16.7168 55.4752 15.2576 132.4544-28.16 175.9232 28.16 0.128 53.6832 1.7408 81.3312-5.4784 55.808-14.592 96.4608-50.0736 121.7792-95.3088 29.6192-52.9664 38.3232-119.2704 25.856-180.992M684.4928 121.344c55.6544 12.6208 183.3472 63.744 260.3008 270.4128 11.52 32.6912 24.0128 61.1584 35.328 65.1264 5.2736 1.8688 12.288 0 19.7632-4.1472h0.0768c15.104-4.864 51.072-28.8512 50.7136-32.768-0.3072-2.8928-6.0416-9.472-8.9088-12.6464 0.9472-1.9456 1.8432-3.8912 2.56-5.8112 0.1024-0.2048 0.1792-0.3328 0.256-0.5888 0.4608-1.2544 0.8448-2.56 1.2032-3.84 0.8704-3.0208 1.408-5.9392 1.536-8.704 1.7152-17.0496-3.1744-33.9968-3.1744-33.9968l-0.1024 0.0512c-7.8336-42.8544-45.312-63.872-62.7968-71.424a37.4784 37.4784 0 0 0-9.2928-3.1488c11.9296-4.0192 29.0304-9.856 42.7264-14.5152a9.344 9.344 0 0 0 5.888-12.032c-10.24-27.0848-27.0336-51.7888-48.3584-73.216h0.0512s-16.1792-19.6608-37.888-43.776c0 0-87.4496-127.9232-151.0144-117.0432-46.6432 8.0384-92.7488 56.9344-113.9968 82.432-7.9104 9.088-12.2368 15.4368-12.2368 15.4368l27.392 4.1728z" />
      </svg>
    </Link>
  );
};

export const AppShortLogo = () => {
  return (
    <Link href="/" className="flex size-8 items-center">
      <svg
        viewBox="0 0 1024 1024"
        version="1.1"
        className="fill-accent-foreground"
      >
        <path d="M1018.7264 711.936c-14.336-70.997333-60.928-127.624533-101.973333-185.0368-46.250667-64.631467-84.992-134.417067-133.376-197.563733-43.810133-57.224533-96.597333-114.414933-172.509867-121.924267-46.592-4.608-91.4432 9.437867-131.754667 32.426667-16.503467 10.069333-95.010133 62.839467-124.3648 167.441066a67.908267 67.908267 0 0 0-17.322666 7.406934l-0.3072 0.170666-0.290134 0.170667a94.3616 94.3616 0 0 0-16.9472 11.349333c-39.0656 28.296533-65.7408 66.645333-83.797333 101.973334l-0.085333 0.0512s-11.997867 21.572267-21.333334 50.2784c-2.218667 6.434133-3.976533 12.168533-5.512533 17.476266C133.751467 542.685867 41.813333 565.691733 0 649.437867c22.084267 3.054933 45.056 3.6864 66.525867 9.642666 71.7824 19.9168 105.915733 78.421333 137.949866 138.6496 17.988267 33.8944 33.877333 69.410133 55.637334 100.7104 63.556267 91.4944 207.291733 113.152 310.2208 46.592-66.304-13.312-136.9088-57.019733-174.114134-105.198933-47.36-61.303467-120.8832-181.640533-162.577066-223.249067-0.512-0.529067-1.092267-0.938667-1.604267-1.450666 12.987733-23.927467 30.958933-47.4624 52.8896-51.6608 20.6336-1.8432 40.106667 4.386133 56.149333 19.592533 16.469333 15.598933 33.314133 36.829867 49.783467 65.570133 79.5648 139.008 178.414933 236.544 343.586133 197.290667 0.034133 0 0.034133 0 0.034134-0.034133-12.1344-64.341333-60.450133-147.968-60.484267-148.0192-35.037867-57.992533-82.858667-134.109867-107.844267-164.9152-33.655467-41.506133-67.8912-87.159467-116.1216-111.957334a155.682133 155.682133 0 0 0-42.666666-14.677333c52.804267-43.588267 132.522667-77.789867 204.219733 19.012267C670.378667 505.173333 783.872 669.986133 826.862933 813.090133c15.957333 53.026133 14.574933 126.5664-26.914133 168.106667 26.897067 0.119467 51.285333 1.672533 77.7216-5.2224 53.2992-13.9264 92.16-47.837867 116.343467-91.0848 28.3136-50.5856 36.625067-113.954133 24.712533-172.9536M654.08 137.335467c53.162667 12.0832 175.189333 60.928 248.7296 258.423466 11.025067 31.232 22.9376 58.436267 33.7408 62.225067 5.0688 1.774933 11.741867 0 18.909867-3.976533h0.068266c14.455467-4.625067 48.810667-27.5456 48.469334-31.2832-0.3072-2.7648-5.7856-9.079467-8.533334-12.117334 0.9216-1.8432 1.757867-3.720533 2.4576-5.5296l0.256-0.5632c0.426667-1.194667 0.802133-2.440533 1.1264-3.6864 0.836267-2.901333 1.348267-5.666133 1.450667-8.2944 1.655467-16.315733-3.003733-32.494933-3.003733-32.494933l-0.119467 0.034133c-7.4752-40.925867-43.298133-61.0304-59.989333-68.232533-2.3552-1.2288-4.2496-1.877333-8.874667-3.0208 11.400533-3.822933 27.716267-9.4208 40.823467-13.858133a8.942933 8.942933 0 0 0 5.597866-11.485867c-9.7792-25.9072-25.8048-49.493333-46.1824-69.973333l0.0512-0.017067s-15.4624-18.773333-36.1984-41.8304c0 0-83.592533-122.231467-144.315733-111.8208-44.561067 7.68-88.610133 54.391467-108.919467 78.7456-7.560533 8.704-11.707733 14.779733-11.707733 14.779733l26.1632 3.976534z" />
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
          <DropdownMenuItem onClick={() => setLocale('zh-CN')}>
            <Check
              data-active={locale === 'zh-CN'}
              className="opacity-0 data-[active=true]:opacity-100"
            />
            简体中文
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
        <DropdownMenuItem onClick={() => setLocale('zh-CN')}>
          <Check
            data-active={locale === 'zh-CN'}
            className="opacity-0 data-[active=true]:opacity-100"
          />
          简体中文
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
    <Link target="_blank" href="https://github.com/apecloud/ApeRAG">
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
