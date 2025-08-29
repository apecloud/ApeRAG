'use client';

import {
  signUpLocalSchema,
  useAppContext,
} from '@/components/providers/app-provider';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
} from '@/components/ui/form';
import { Input } from '@/components/ui/input';

import { zodResolver } from '@hookform/resolvers/zod';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { useCallback } from 'react';
import { useForm } from 'react-hook-form';
import * as z from 'zod';

export function SignUpForm() {
  const searchParams = useSearchParams();
  const { signUp } = useAppContext();
  const page_auth = useTranslations('page_auth');

  const callbackUrl = searchParams.get('callbackUrl') || '/';
  const form = useForm<z.infer<typeof signUpLocalSchema>>({
    resolver: zodResolver(signUpLocalSchema),
    defaultValues: {
      username: '',
      password: '',
      email: '',
    },
  });

  const handleSignUpLocal = useCallback(
    async (payload: z.infer<typeof signUpLocalSchema>) => {
      await signUp({
        data: payload,
        redirectTo: callbackUrl,
      });
    },
    [callbackUrl, signUp],
  );

  return (
    <div className="flex flex-col gap-6">
      <Card className="bg-card/50">
        <CardContent>
          <div className="mb-8 text-center text-xl font-bold">
            {page_auth('register_an_account')}
          </div>
          <Form {...form}>
            <form
              onSubmit={form.handleSubmit(handleSignUpLocal)}
              className="grid gap-6"
            >
              <FormField
                control={form.control}
                name="username"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>{page_auth('username')}</FormLabel>
                    <FormControl>
                      <Input
                        {...field}
                        placeholder={page_auth('username_placeholder')}
                      />
                    </FormControl>
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="email"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>{page_auth('email')}</FormLabel>
                    <FormControl>
                      <Input
                        {...field}
                        placeholder={page_auth('email_placeholder')}
                      />
                    </FormControl>
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="password"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>{page_auth('password')}</FormLabel>
                    <FormControl>
                      <Input
                        type="password"
                        {...field}
                        placeholder={page_auth('password_placeholder')}
                      />
                    </FormControl>
                  </FormItem>
                )}
              />

              <Button type="submit" className="w-full">
                {page_auth('signup')}
              </Button>

              <div className="text-center text-sm">
                {page_auth('already_hava_an_account')}
                <Link
                  href={`/auth/signin?callbackUrl=${encodeURIComponent(callbackUrl)}`}
                  className="underline underline-offset-4"
                >
                  {page_auth('signin')}
                </Link>
              </div>
            </form>
          </Form>
        </CardContent>
      </Card>
      {/* <div className="text-muted-foreground *:[a]:hover:text-primary text-center text-xs text-balance *:[a]:underline *:[a]:underline-offset-4">
        By clicking continue, you agree to our{' '}
        <Link href="#">Terms of Service</Link> and{' '}
        <Link href="#">Privacy Policy</Link>.
      </div> */}
    </div>
  );
}
