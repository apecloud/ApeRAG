'use client';

import { useAppContext } from '@/components/providers/app-provider';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { useTranslations } from 'next-intl';

import Link from 'next/link';

export default function Page() {
  const { signIn } = useAppContext();
  const page_auth = useTranslations('page_auth');

  return (
    <div className="flex flex-col gap-6">
      <Card className="bg-card/50">
        <CardHeader className="text-center">
          <CardTitle className="text-xl">
            {page_auth('verify_request_title')}
          </CardTitle>
          <CardDescription>
            {page_auth('verify_request_description')}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="mt-10 flex items-center justify-center gap-x-6">
            <Link href="/">
              <Button>{page_auth('go_back_home')}</Button>
            </Link>
            <Button
              variant="outline"
              onClick={() => signIn({ redirectTo: '/' })}
            >
              <div className="grid flex-1 text-left text-sm leading-tight">
                {page_auth('sign_in_again')}
              </div>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}