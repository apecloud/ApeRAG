'use client';

import { Button } from '@/components/ui/button';
import { useGlobalContext } from '@/hooks/use-global-context';
import Link from 'next/link';
import { useParams, useSearchParams } from 'next/navigation';
import { useEffect, useState } from 'react';

export default function Page() {
  const { signIn } = useGlobalContext();
  const { provider } = useParams();
  const searchParams = useSearchParams();
  const error = searchParams.get('error');
  const code = searchParams.get('code') || '';
  const state = searchParams.get('state') || '';

  const [tips, setTips] = useState<string>('Processing OAuth login...');

  useEffect(() => {
    if (!code || !state) return;
    const callbackUrl = `/api/v1/auth/${provider}/callback?code=${encodeURIComponent(code)}&state=${encodeURIComponent(state)}`;
    fetch(callbackUrl, {
      method: 'GET',
      credentials: 'include',
      redirect: 'manual',
    })
      .then((res) => {
        if (res.ok) {
          setTimeout(() => {
            window.location.href = '/';
          }, 1000);
          return;
        }
        if (res.status === 500) {
          setTips('oauth_failed');
        }
      })
      .catch((err) => {
        console.log(err);
      });
  }, [code, provider, state]);

  useEffect(() => {
    if (error) {
      setTips('oauth_failed');
      return;
    }

    if (!code || !state) {
      setTips('oauth_invalid');
    }
  }, [error, code, state]);

  return (
    <div className="bg-accent/10 flex flex-col gap-6 rounded-md px-4 py-8 text-center">
      <div className="text-xl">Authentication</div>
      <div className="text-muted-foreground text-sm">{tips}</div>
      <div className="mt-10 flex items-center justify-center gap-x-6">
        <Link href="/">
          <Button>Go back home</Button>
        </Link>
        <Button variant="outline" onClick={() => signIn({ redirectTo: '/' })}>
          <div className="grid flex-1 text-left text-sm leading-tight">
            Sign in again
          </div>
        </Button>
      </div>
    </div>
  );
}
