'use client';

import { useAppContext } from '@/components/providers/app-provider';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { LoaderCircle, ShieldAlert, ShieldPlus } from 'lucide-react';
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

  const content = useMemo(() => {
    if (loading) {
      return (
        <>
          <LoaderCircle className="size-12 animate-spin opacity-50" />
          <div className="text-muted-foreground text-sm">
            Processing OAuth login...
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
        <div className="text-muted-foreground text-sm">OAuth successful!</div>
        <div className="text-muted-foreground text-sm">
          the system will automatically redirect.
        </div>
      </>
    );
  }, [loading, tips]);

  useEffect(() => {
    if (!code || !state) return;
    const callbackUrl = `${process.env.NEXT_PUBLIC_BASE_PATH || ''}${process.env.NEXT_PUBLIC_API_SERVER_BASE_PATH}/auth/${provider}/callback?code=${encodeURIComponent(code)}&state=${encodeURIComponent(state)}`;
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
        setTips('OAuth verification failed.');
      })
      .catch((err) => {
        console.log(err);
        setTips('An unexpected error occurred');
      })
      .finally(() => {
        setLoading(false);
      });
  }, [code, provider, state]);

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
        <div className="text-center text-xl font-bold">Authentication</div>

        <div className="flex flex-col items-center justify-center gap-2 text-center">
          {content}
        </div>

        <div className="flex items-center justify-center gap-x-6">
          <Link href="/">
            <Button>Go back home</Button>
          </Link>
          <Button variant="outline" onClick={() => signIn({ redirectTo: '/' })}>
            <div className="grid flex-1 text-left text-sm leading-tight">
              Retry
            </div>
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
