import { AppLocaleDropdownMenu, AppLogo } from '@/components/app-topbar';
import { Button } from '@/components/ui/button';
import {
  ArrowRight,
  BookOpenText,
  Boxes,
  Check,
  DatabaseZap,
  GitBranch,
  Mail,
  Network,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';
import { getTranslations } from 'next-intl/server';
import Link from 'next/link';

const navLinks = [
  { href: '#agent', key: 'nav.agent' },
  { href: '#graph', key: 'nav.graph' },
  { href: '#deployment', key: 'nav.deployment' },
] as const;

const capabilityKeys = ['hybrid', 'graph', 'agent', 'deployment'] as const;
const traceSteps = [
  'question',
  'graph',
  'manual',
  'compare',
  'answer',
] as const;
const agentFeatureKeys = ['runtime', 'mcp', 'audit'] as const;
const deploymentFeatureKeys = ['private', 'models', 'management'] as const;

type LandingTranslations = Awaited<ReturnType<typeof getTranslations>>;

export default async function Home() {
  const t = await getTranslations('page_landing');

  return (
    <main className="bg-background text-foreground min-h-[100dvh] overflow-hidden">
      <LandingNav t={t} />
      <section className="relative mx-auto grid max-w-7xl gap-12 px-6 pt-28 pb-20 md:grid-cols-[1.02fr_0.98fr] md:px-10 md:pt-36 lg:gap-16">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-[44rem] opacity-90"
          style={{
            backgroundImage:
              'radial-gradient(circle at 76% 18%, color-mix(in oklab, var(--primary) 14%, transparent), transparent 32rem), radial-gradient(circle at 12% 32%, color-mix(in oklab, var(--accent-ink) 8%, transparent), transparent 26rem)',
          }}
        />
        <div className="flex flex-col justify-center">
          <div className="bg-card text-muted-foreground mb-7 inline-flex w-fit items-center rounded-full border px-3 py-1 text-xs shadow-xs">
            <span>{t('hero.badge_text')}</span>
          </div>
          <h1 className="max-w-3xl font-serif text-5xl leading-[1.02] font-normal tracking-[-0.045em] text-balance md:text-6xl lg:text-7xl">
            {t('hero.title')}{' '}
            <span className="text-primary">{t('hero.title_accent')}</span>
          </h1>
          <p className="text-muted-foreground mt-7 max-w-xl text-base leading-8 md:text-lg">
            {t('hero.description')}
          </p>
          <div className="mt-9 flex flex-col gap-3 sm:flex-row">
            <Button
              asChild
              size="lg"
              className="h-11 rounded-full px-6 active:scale-[0.98]"
            >
              <Link href="/workspace/collections">
                {t('hero.primary_cta')}
                <ArrowRight className="size-4" />
              </Link>
            </Button>
            <Button
              asChild
              variant="outline"
              size="lg"
              className="h-11 rounded-full px-6 active:scale-[0.98]"
            >
              <Link href="/marketplace">{t('hero.secondary_cta')}</Link>
            </Button>
          </div>
          <div className="text-muted-foreground mt-9 grid gap-3 text-sm sm:grid-cols-2">
            {(['private', 'models'] as const).map((key) => (
              <div key={key} className="flex items-center gap-2">
                <span className="bg-accent-soft text-accent-ink grid size-5 place-items-center rounded-full">
                  <Check className="size-3" />
                </span>
                {t(`hero.points.${key}`)}
              </div>
            ))}
          </div>
        </div>
        <HeroAgentTrace t={t} />
      </section>

      <section className="mx-auto max-w-7xl px-6 pb-20 md:px-10">
        <div className="bg-card grid overflow-hidden rounded-xl border shadow-sm md:grid-cols-4">
          {capabilityKeys.map((key, index) => (
            <div
              key={key}
              className="border-b p-6 last:border-b-0 md:border-r md:border-b-0 last:md:border-r-0"
            >
              <div className="text-primary font-mono text-[11px] tracking-[0.16em] uppercase">
                0{index + 1}
              </div>
              <div className="mt-5 text-lg font-medium tracking-tight">
                {t(`capabilities.${key}.title`)}
              </div>
              <p className="text-muted-foreground mt-2 text-sm leading-6">
                {t(`capabilities.${key}.description`)}
              </p>
            </div>
          ))}
        </div>
      </section>

      <section
        id="agent"
        className="mx-auto grid max-w-7xl gap-12 px-6 py-20 md:grid-cols-[0.88fr_1.12fr] md:px-10 lg:gap-20"
      >
        <SectionIntro
          eyebrow={t('agent.eyebrow')}
          title={t('agent.title')}
          description={t('agent.description')}
        />
        <div className="grid gap-4 md:grid-cols-2">
          {agentFeatureKeys.map((key, index) => (
            <FeaturePanel
              key={key}
              index={index + 1}
              title={t(`agent.features.${key}.title`)}
              description={t(`agent.features.${key}.description`)}
            />
          ))}
        </div>
      </section>

      <section
        id="graph"
        className="mx-auto grid max-w-7xl gap-12 px-6 py-20 md:grid-cols-[1.12fr_0.88fr] md:px-10 lg:gap-20"
      >
        <GraphVisual t={t} />
        <SectionIntro
          eyebrow={t('graph.eyebrow')}
          title={t('graph.title')}
          description={t('graph.description')}
          align="right"
        />
      </section>

      <section
        id="deployment"
        className="mx-auto max-w-7xl px-6 py-20 md:px-10"
      >
        <div className="bg-card grid gap-8 rounded-xl border p-6 shadow-sm md:grid-cols-[0.9fr_1.1fr] md:p-10">
          <SectionIntro
            eyebrow={t('deployment.eyebrow')}
            title={t('deployment.title')}
            description={t('deployment.description')}
          />
          <div className="grid gap-4 sm:grid-cols-3">
            {deploymentFeatureKeys.map((key) => (
              <div key={key} className="bg-background rounded-xl border p-5">
                <div className="bg-accent-soft text-accent-ink mb-6 grid size-10 place-items-center rounded-full">
                  {key === 'private' ? (
                    <ShieldCheck className="size-5" />
                  ) : key === 'models' ? (
                    <DatabaseZap className="size-5" />
                  ) : (
                    <Boxes className="size-5" />
                  )}
                </div>
                <div className="font-medium tracking-tight">
                  {t(`deployment.features.${key}.title`)}
                </div>
                <p className="text-muted-foreground mt-2 text-sm leading-6">
                  {t(`deployment.features.${key}.description`)}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-6 py-20 md:px-10">
        <div className="bg-foreground text-background relative overflow-hidden rounded-xl border p-8 shadow-sm md:p-12">
          <div className="max-w-2xl">
            <div className="text-background/60 font-mono text-xs tracking-[0.18em] uppercase">
              {t('cta.eyebrow')}
            </div>
            <h2 className="mt-5 font-serif text-4xl leading-tight font-normal tracking-[-0.035em] md:text-5xl">
              {t('cta.title')}
            </h2>
            <p className="text-background/70 mt-5 text-base leading-7">
              {t('cta.description')}
            </p>
          </div>
          <div className="mt-8 flex flex-col gap-3 sm:flex-row">
            <Button
              asChild
              size="lg"
              className="bg-background text-foreground hover:bg-background/90 h-11 rounded-full px-6 active:scale-[0.98]"
            >
              <Link href="/workspace/collections">{t('cta.primary_cta')}</Link>
            </Button>
            <Button
              asChild
              size="lg"
              variant="outline"
              className="border-background/20 text-background hover:bg-background/10 hover:text-background h-11 rounded-full bg-transparent px-6 active:scale-[0.98]"
            >
              <a href="mailto:sailwebs@apecloud.com">
                <Mail className="size-4" />
                {t('cta.secondary_cta')}
              </a>
            </Button>
          </div>
        </div>
      </section>
    </main>
  );
}

function LandingNav({ t }: { t: LandingTranslations }) {
  return (
    <header className="bg-background/85 fixed inset-x-0 top-0 z-40 border-b backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-7xl items-center gap-8 px-6 md:px-10">
        <AppLogo />
        <nav className="text-muted-foreground hidden items-center gap-7 text-sm md:flex">
          {navLinks.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="hover:text-foreground transition-colors"
            >
              {t(item.key)}
            </Link>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-2">
          <AppLocaleDropdownMenu />
          <Button asChild variant="ghost" size="sm" className="rounded-full">
            <Link href="/auth/signin">{t('nav.signin')}</Link>
          </Button>
        </div>
      </div>
    </header>
  );
}

function HeroAgentTrace({ t }: { t: LandingTranslations }) {
  return (
    <div className="relative self-center">
      <div
        aria-hidden="true"
        className="absolute -inset-8 -z-10 rounded-[2rem] opacity-80"
        style={{
          backgroundImage:
            'radial-gradient(circle at 30% 20%, color-mix(in oklab, var(--primary) 12%, transparent), transparent 22rem), radial-gradient(circle at 82% 82%, color-mix(in oklab, var(--accent-ink) 8%, transparent), transparent 18rem)',
        }}
      />
      <div className="bg-card overflow-hidden rounded-xl border shadow-sm">
        <div className="flex items-center gap-3 border-b px-5 py-4">
          <div className="flex gap-1.5" aria-hidden="true">
            <span className="bg-accent-soft size-2.5 rounded-full" />
            <span className="bg-secondary size-2.5 rounded-full" />
            <span className="bg-muted size-2.5 rounded-full" />
          </div>
          <span className="text-muted-foreground font-mono text-xs">
            agent · collections/smt-manuals
          </span>
          <span className="text-primary ml-auto inline-flex items-center gap-2 font-mono text-[11px] uppercase">
            <span className="bg-primary size-1.5 rounded-full" />
            {t('hero_trace.status')}
          </span>
        </div>
        <div className="px-6 py-5">
          <div className="text-muted-foreground font-mono text-[11px] tracking-[0.14em] uppercase">
            {t('hero_trace.user')}
          </div>
          <p className="mt-2 text-sm leading-6">{t('hero_trace.prompt')}</p>
        </div>
        <div className="border-t p-4">
          <div className="bg-muted border-border/70 overflow-hidden rounded-xl border">
            <div className="text-muted-foreground flex items-center gap-2 px-3.5 py-2.5">
              <Sparkles className="text-primary size-3" />
              <span className="font-mono text-[10.5px] tracking-[0.08em] uppercase">
                {t('hero_trace.title')}
              </span>
              <span className="text-muted-foreground/80 ml-2 truncate text-[11px]">
                {t('hero_trace.meta')}
              </span>
            </div>
            <div className="border-border/70 border-t px-4 py-3.5">
              <div className="relative flex flex-col gap-3.5">
                <div className="bg-border absolute top-3 bottom-3 left-[11px] w-px" />
                {traceSteps.map((key, index) => (
                  <div
                    key={key}
                    className="relative grid grid-cols-[1.5rem_1fr] gap-3"
                  >
                    <div className="bg-card text-primary z-10 grid size-6 place-items-center rounded-full shadow-xs">
                      {index === 0 ? (
                        <Sparkles className="size-3.5" />
                      ) : index === 1 ? (
                        <Network className="size-3.5" />
                      ) : index === 2 ? (
                        <BookOpenText className="size-3.5" />
                      ) : index === 3 ? (
                        <GitBranch className="size-3.5" />
                      ) : (
                        <Check className="size-3.5" />
                      )}
                    </div>
                    <div className="min-w-0">
                      <div className="text-muted-foreground font-mono text-[10px] tracking-[0.14em] uppercase">
                        {t(`hero_trace.steps.${key}.label`)}
                      </div>
                      <p className="text-foreground/85 mt-1 text-sm leading-6">
                        {t(`hero_trace.steps.${key}.text`)}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function SectionIntro({
  eyebrow,
  title,
  description,
  align = 'left',
}: {
  eyebrow: string;
  title: string;
  description: string;
  align?: 'left' | 'right';
}) {
  return (
    <div className={align === 'right' ? 'md:pl-8' : ''}>
      <div className="text-primary font-mono text-xs tracking-[0.18em] uppercase">
        {eyebrow}
      </div>
      <h2 className="mt-5 max-w-xl font-serif text-4xl leading-[1.08] font-normal tracking-[-0.035em] md:text-5xl">
        {title}
      </h2>
      <p className="text-muted-foreground mt-6 max-w-xl text-base leading-7">
        {description}
      </p>
    </div>
  );
}

function FeaturePanel({
  index,
  title,
  description,
}: {
  index: number;
  title: string;
  description: string;
}) {
  return (
    <div
      className={
        index === 1
          ? 'bg-card rounded-xl border p-6 shadow-sm md:col-span-2'
          : 'bg-card rounded-xl border p-6 shadow-sm'
      }
    >
      <div className="text-primary font-mono text-[11px] tracking-[0.16em] uppercase">
        0{index}
      </div>
      <div className="mt-8 text-lg font-medium tracking-tight">{title}</div>
      <p className="text-muted-foreground mt-2 max-w-xl text-sm leading-6">
        {description}
      </p>
    </div>
  );
}

function GraphVisual({ t }: { t: LandingTranslations }) {
  const nodes = [
    [
      '50%',
      '48%',
      'h-18 w-18 bg-primary text-primary-foreground',
      t('graph.nodes.center'),
    ],
    ['24%', '22%', 'h-8 w-8 bg-chart-2 text-background', t('graph.nodes.org')],
    [
      '76%',
      '25%',
      'h-10 w-10 bg-chart-4 text-background',
      t('graph.nodes.product'),
    ],
    [
      '20%',
      '72%',
      'h-9 w-9 bg-chart-5 text-background',
      t('graph.nodes.event'),
    ],
    [
      '76%',
      '76%',
      'h-8 w-8 bg-chart-1 text-background',
      t('graph.nodes.person'),
    ],
    [
      '48%',
      '16%',
      'h-7 w-7 bg-chart-3 text-background',
      t('graph.nodes.concept'),
    ],
  ] as const;

  return (
    <div className="bg-card relative min-h-[25rem] rounded-xl border p-5 shadow-sm">
      <svg
        className="absolute inset-5 h-[calc(100%-2.5rem)] w-[calc(100%-2.5rem)]"
        aria-hidden="true"
      >
        <line
          x1="50%"
          y1="48%"
          x2="24%"
          y2="22%"
          className="stroke-border"
          strokeWidth="1"
        />
        <line
          x1="50%"
          y1="48%"
          x2="76%"
          y2="25%"
          className="stroke-border"
          strokeWidth="1"
        />
        <line
          x1="50%"
          y1="48%"
          x2="20%"
          y2="72%"
          className="stroke-border"
          strokeWidth="1"
        />
        <line
          x1="50%"
          y1="48%"
          x2="76%"
          y2="76%"
          className="stroke-border"
          strokeWidth="1"
        />
        <line
          x1="50%"
          y1="48%"
          x2="48%"
          y2="16%"
          className="stroke-border"
          strokeWidth="1"
        />
      </svg>
      {nodes.map(([left, top, className, label]) => (
        <div
          key={label}
          className="absolute -translate-x-1/2 -translate-y-1/2"
          style={{ left, top }}
        >
          <div
            className={`${className} border-card grid place-items-center rounded-full border-4 font-mono text-[10px] shadow-sm`}
          >
            {label}
          </div>
        </div>
      ))}
      <div className="bg-background/90 absolute bottom-5 left-5 rounded-xl border p-4 shadow-sm backdrop-blur">
        <div className="text-muted-foreground font-mono text-[11px] tracking-[0.16em] uppercase">
          {t('graph.legend')}
        </div>
        <div className="text-muted-foreground mt-3 grid grid-cols-2 gap-x-5 gap-y-2 text-xs">
          {['person', 'org', 'product', 'event'].map((item, index) => (
            <div key={item} className="flex items-center gap-2">
              <span
                className={`size-2 rounded-full ${
                  index === 0
                    ? 'bg-chart-1'
                    : index === 1
                      ? 'bg-chart-2'
                      : index === 2
                        ? 'bg-chart-4'
                        : 'bg-chart-5'
                }`}
              />
              {t(`graph.legend_items.${item}`)}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
