'use client';

import { useAppContext } from '@/components/providers/app-provider';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';

import { MailCheck } from 'lucide-react';
import { useTranslations } from 'next-intl';
import Link from 'next/link';

export default function Page() {
  const { signIn } = useAppContext();
  const page_auth = useTranslations('page_auth');

  return (
    <div className="flex flex-col gap-6">
      <Card className="border-border/80 bg-card/95 shadow-sm">
        <CardContent className="px-6 py-7 text-center md:px-7">
          <div className="bg-accent-soft text-accent-ink mx-auto grid size-12 place-items-center rounded-full">
            <MailCheck className="size-6" />
          </div>
          <h1 className="mt-5 font-serif text-4xl leading-tight font-normal tracking-[-0.035em]">
            {page_auth('check_email_title')}
          </h1>
          <p className="text-muted-foreground mt-3 text-sm leading-6">
            {page_auth('check_email_description')}
          </p>
          <div className="mt-8 flex flex-col items-center justify-center gap-3 sm:flex-row">
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
              {page_auth('signin_again')}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
