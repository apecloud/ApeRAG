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

import Link from 'next/link';

export default function Page() {
  const { signIn } = useAppContext();
  return (
    <div className="flex flex-col gap-6">
      <Card className="bg-card/50">
        <CardHeader className="text-center">
          <CardTitle className="text-xl">Check your email</CardTitle>
          <CardDescription>
            A sign in link has been sent to your email address.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="mt-10 flex items-center justify-center gap-x-6">
            <Link href="/">
              <Button>Go back home</Button>
            </Link>
            <Button
              variant="outline"
              onClick={() => signIn({ redirectTo: '/' })}
            >
              <div className="grid flex-1 text-left text-sm leading-tight">
                Sign in again
              </div>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
