'use client';

import { useAppContext } from '@/components/providers/app-provider';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { LoaderCircle, ShieldAlert, ShieldPlus } from 'lucide-react';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { useParams, useSearchParams } from 'next/navigation';
import { useEffect, useMemo, useState } from 'react';

export default function Page() {
  const { signIn } = useAppContext();
  const { provider } = useParams();
  const searchParams = useSearchParams();
  const [loading, setLoading] = useState<boolean>(true);
  const [tips, setTips] = useState<string>();

  const error = searchParams.get('error');
  const code = searchParams.get('code') || '';
  const state = searchParams.get('state') || '';
  const page_auth = useTranslations('page_auth');
  const content = useMemo(() => {
    if (loading) {
      return (
        <>
          <div className="bg-accent-soft text-accent-ink grid size-12 place-items-center rounded-full">
            <LoaderCircle className="size-6 animate-spin" />
          </div>
          <div className="text-muted-foreground text-sm">
            {page_auth('processing_oauth_login')}
          </div>
        </>
      );
    }
    if (tips) {
      return (
        <>
          <div className="bg-destructive/10 text-destructive grid size-12 place-items-center rounded-full">
            <ShieldAlert className="size-6" />
          </div>
          <div className="text-muted-foreground text-sm">{tips}</div>
        </>
      );
    }
    return (
      <>
        <div className="bg-accent-soft text-accent-ink grid size-12 place-items-center rounded-full">
          <ShieldPlus className="size-6" />
        </div>
        <div className="text-muted-foreground text-sm">
          {page_auth('oauth_successful')}
        </div>
        <div className="text-muted-foreground text-sm">
          {page_auth('the_system_will_automatically_redirect')}
        </div>
      </>
    );
  }, [loading, page_auth, tips]);

  useEffect(() => {
    if (!code || !state) return;
    const callbackUrl = `${process.env.NEXT_PUBLIC_BASE_PATH || ''}/api/v2/auth/${provider}/callback?code=${encodeURIComponent(code)}&state=${encodeURIComponent(state)}`;
    fetch(callbackUrl, {
      method: 'GET',
      credentials: 'include',
      redirect: 'manual',
    })
      .then((res) => {
        if (res.status >= 200) {
          setTimeout(() => {
            window.location.href = '/workspace';
          }, 300);
          return;
        }
        setTips(page_auth('oauth_verification_failed'));
      })
      .catch((err) => {
        console.log(err);
        setTips(page_auth('an_unexpected_error_occurred'));
      })
      .finally(() => {
        setLoading(false);
      });
  }, [code, page_auth, provider, state]);

  useEffect(() => {
    if (error) {
      setTips(error);
      return;
    }

    if (!code || !state) {
      setTips('Invalid parameter');
    }
  }, [error, code, state]);

  return (
    <Card className="border-border/80 bg-card/95 shadow-sm">
      <CardContent className="flex flex-col gap-9 px-6 py-7 md:px-7">
        <div className="text-center">
          <div className="text-primary font-mono text-xs tracking-[0.18em] uppercase">
            {page_auth('oauth_eyebrow')}
          </div>
          <h1 className="mt-3 font-serif text-4xl leading-tight font-normal tracking-[-0.035em]">
            {page_auth('authentication')}
          </h1>
        </div>

        <div className="flex flex-col items-center justify-center gap-3 text-center">
          {content}
        </div>

        <div className="flex flex-col items-center justify-center gap-3 sm:flex-row">
          <Link href="/">
            <Button className="rounded-full">
              {page_auth('go_back_home')}
            </Button>
          </Link>
          <Button
            variant="outline"
            className="rounded-full"
            onClick={() => signIn({ redirectTo: '/' })}
          >
            {page_auth('retry')}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
