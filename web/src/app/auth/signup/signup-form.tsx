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
  FormMessage,
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
      <Card className="border-border/80 bg-card/95 shadow-sm">
        <CardContent className="px-6 py-7 md:px-7">
          <div className="mb-7">
            <div className="text-primary font-mono text-xs tracking-[0.18em] uppercase">
              {page_auth('signup_eyebrow')}
            </div>
            <h1 className="mt-3 font-serif text-4xl leading-tight font-normal tracking-[-0.035em]">
              {page_auth('register_an_account')}
            </h1>
            <p className="text-muted-foreground mt-2 text-sm leading-6">
              {page_auth('signup_description')}
            </p>
          </div>
          <Form {...form}>
            <form
              onSubmit={form.handleSubmit(handleSignUpLocal)}
              className="grid gap-5"
            >
              <FormField
                control={form.control}
                name="username"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-foreground text-xs font-medium">
                      {page_auth('username')}
                    </FormLabel>
                    <FormControl>
                      <Input
                        {...field}
                        className="bg-background h-11 rounded-lg"
                        placeholder={page_auth('username_placeholder')}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="email"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-foreground text-xs font-medium">
                      {page_auth('email')}
                    </FormLabel>
                    <FormControl>
                      <Input
                        {...field}
                        className="bg-background h-11 rounded-lg"
                        placeholder={page_auth('email_placeholder')}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="password"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-foreground text-xs font-medium">
                      {page_auth('password')}
                    </FormLabel>
                    <FormControl>
                      <Input
                        type="password"
                        {...field}
                        className="bg-background h-11 rounded-lg"
                        placeholder={page_auth('password_placeholder')}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <Button
                type="submit"
                className="h-11 w-full rounded-full active:scale-[0.98]"
              >
                {page_auth('signup')}
              </Button>

              <div className="text-muted-foreground text-center text-sm">
                {page_auth('already_hava_an_account')}
                <Link
                  href={`/auth/signin?callbackUrl=${encodeURIComponent(callbackUrl)}`}
                  className="text-primary font-medium underline-offset-4 hover:underline"
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
