import { AppLogo } from '@/components/app-topbar';
import { getTranslations } from 'next-intl/server';

export default async function AuthLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const t = await getTranslations('page_auth');

  return (
    <main className="bg-background text-foreground grid min-h-[100dvh] lg:grid-cols-[1.05fr_0.95fr]">
      <section className="bg-secondary/65 relative hidden overflow-hidden border-r p-10 lg:flex lg:flex-col xl:p-14">
        <AppLogo />
        <div className="relative z-10 flex flex-1 flex-col justify-center">
          <div className="max-w-xl">
            <div className="text-primary font-mono text-xs tracking-[0.18em] uppercase">
              {t('auth_hero_eyebrow')}
            </div>
            <blockquote className="mt-6 font-serif text-4xl leading-tight font-normal tracking-[-0.035em] text-balance">
              {t('auth_hero_quote')}
            </blockquote>
            <p className="text-muted-foreground mt-6 max-w-md text-sm leading-7">
              {t('auth_hero_description')}
            </p>
          </div>
        </div>
        <AuthGraphFlourish />
      </section>

      <section className="relative flex min-h-[100dvh] flex-col px-6 py-8 md:px-10 lg:px-14">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 -z-10"
          style={{
            backgroundImage:
              'radial-gradient(circle at 85% 12%, color-mix(in oklab, var(--primary) 9%, transparent), transparent 24rem), radial-gradient(circle at 12% 85%, color-mix(in oklab, var(--accent-ink) 6%, transparent), transparent 22rem)',
          }}
        />
        <div className="flex items-center justify-between lg:hidden">
          <AppLogo />
        </div>
        <div className="flex flex-1 items-center justify-center py-12">
          <div className="w-full max-w-md">{children}</div>
        </div>
        <p className="text-muted-foreground mx-auto max-w-md text-center text-xs leading-5">
          {t('auth_footer')}
        </p>
      </section>
    </main>
  );
}

function AuthGraphFlourish() {
  const nodes = [
    ['50%', '50%', 'size-15 bg-primary'],
    ['28%', '28%', 'size-8 bg-chart-2'],
    ['75%', '30%', 'size-9 bg-chart-4'],
    ['24%', '72%', 'size-7 bg-chart-5'],
    ['78%', '70%', 'size-8 bg-chart-1'],
    ['52%', '18%', 'size-6 bg-chart-3'],
  ] as const;

  return (
    <div className="pointer-events-none absolute right-[-5rem] bottom-[-4rem] h-[24rem] w-[30rem] opacity-80">
      <div
        aria-hidden="true"
        className="absolute inset-0 rounded-full"
        style={{
          backgroundImage:
            'radial-gradient(circle, color-mix(in oklab, var(--primary) 13%, transparent), transparent 60%)',
        }}
      />
      <svg className="absolute inset-0 size-full" aria-hidden="true">
        <line x1="50%" y1="50%" x2="28%" y2="28%" className="stroke-border" />
        <line x1="50%" y1="50%" x2="75%" y2="30%" className="stroke-border" />
        <line x1="50%" y1="50%" x2="24%" y2="72%" className="stroke-border" />
        <line x1="50%" y1="50%" x2="78%" y2="70%" className="stroke-border" />
        <line x1="50%" y1="50%" x2="52%" y2="18%" className="stroke-border" />
      </svg>
      {nodes.map(([left, top, className]) => (
        <span
          key={`${left}-${top}`}
          className={`border-secondary absolute -translate-x-1/2 -translate-y-1/2 rounded-full border-4 ${className}`}
          style={{ left, top }}
        />
      ))}
    </div>
  );
}
