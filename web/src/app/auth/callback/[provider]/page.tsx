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
          <LoaderCircle className="size-12 animate-spin opacity-50" />
          <div className="text-muted-foreground text-sm">
            {page_auth('processing_oauth_login')}
          </div>
        </>
      );
    }
    if (tips) {
      return (
        <>
          <ShieldAlert className="size-12" />
          <div className="text-muted-foreground text-sm">{tips}</div>
        </>
      );
    }
    return (
      <>
        <ShieldPlus className="size-12" />
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
    const callbackUrl = `${process.env.NEXT_PUBLIC_BASE_PATH || ''}/api/v1/auth/${provider}/callback?code=${encodeURIComponent(code)}&state=${encodeURIComponent(state)}`;
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
    <Card className="bg-card/50">
      <CardContent className="flex flex-col gap-12">
        <div className="text-center text-xl font-bold">
          {page_auth('authentication')}
        </div>

        <div className="flex flex-col items-center justify-center gap-2 text-center">
          {content}
        </div>

        <div className="flex items-center justify-center gap-x-6">
          <Link href="/">
            <Button>{page_auth('go_back_home')}</Button>
          </Link>
          <Button variant="outline" onClick={() => signIn({ redirectTo: '/' })}>
            <div className="grid flex-1 text-left text-sm leading-tight">
              {page_auth('retry')}
            </div>
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
