'use client';

import {
  signInLocalSchema,
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
import { cn } from '@/lib/utils';
import { zodResolver } from '@hookform/resolvers/zod';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { useCallback, useMemo } from 'react';
import { useForm } from 'react-hook-form';
import { FaGithub, FaGoogle } from 'react-icons/fa6';
import * as z from 'zod';

export function SignInForm({
  className,
  methods,
  ...props
}: React.ComponentProps<'div'> & {
  methods: string[];
}) {
  const searchParams = useSearchParams();
  const redirectTo = searchParams.get('callbackUrl') || '/';
  const { signIn } = useAppContext();
  const page_auth = useTranslations('page_auth');
  const form = useForm<z.infer<typeof signInLocalSchema>>({
    resolver: zodResolver(signInLocalSchema),
    defaultValues: {
      username: '',
      password: '',
    },
  });

  const hasSocialLogin = useMemo(() => {
    return methods.some((method) => ['google', 'github'].includes(method));
  }, [methods]);

  const hasSocialGithubLogin = useMemo(() => {
    return methods.some((method) => method === 'github');
  }, [methods]);

  const hasSocialGoogleLogin = useMemo(() => {
    return methods.some((method) => method === 'google');
  }, [methods]);

  const handleSignInLocal = useCallback(
    async (payload: z.infer<typeof signInLocalSchema>) => {
      await signIn({
        type: 'local',
        data: payload,
        redirectTo,
      });
    },
    [redirectTo, signIn],
  );

  return (
    <div className={cn('flex flex-col gap-6', className)} {...props}>
      <Card className="border-border/80 bg-card/95 shadow-sm">
        <CardContent className="px-6 py-7 md:px-7">
          <div className="mb-7">
            <div className="text-primary font-mono text-xs tracking-[0.18em] uppercase">
              {page_auth('signin_eyebrow')}
            </div>
            <h1 className="mt-3 font-serif text-4xl leading-tight font-normal tracking-[-0.035em]">
              {page_auth('welcome_back')}
            </h1>
            <p className="text-muted-foreground mt-2 text-sm leading-6">
              {page_auth('signin_description')}
            </p>
          </div>
          {hasSocialLogin && (
            <div className="mb-6 grid gap-3">
              <div className="text-muted-foreground text-center text-sm">
                {page_auth('login_in_with_a_third_party_account')}
              </div>
              {hasSocialGithubLogin && (
                <Button
                  variant="outline"
                  className="h-10 w-full rounded-lg active:scale-[0.98]"
                  onClick={() => signIn({ type: 'github', redirectTo })}
                >
                  <FaGithub />
                  {page_auth('login_with_github')}
                </Button>
              )}
              {hasSocialGoogleLogin && (
                <Button
                  variant="outline"
                  className="h-10 w-full rounded-lg active:scale-[0.98]"
                  onClick={() => signIn({ type: 'google', redirectTo })}
                >
                  <FaGoogle />
                  {page_auth('login_with_google')}
                </Button>
              )}

              <div className="after:border-border relative text-center text-sm after:absolute after:inset-0 after:top-1/2 after:z-0 after:flex after:items-center after:border-t">
                <span className="bg-card text-muted-foreground relative z-10 px-3 font-mono text-[11px] tracking-[0.12em] uppercase">
                  {page_auth('or_continue_with')}
                </span>
              </div>
            </div>
          )}
          <Form {...form}>
            <form
              onSubmit={form.handleSubmit(handleSignInLocal)}
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
                name="password"
                render={({ field }) => (
                  <FormItem>
                    <div className="flex justify-between">
                      <FormLabel className="text-foreground text-xs font-medium">
                        {page_auth('password')}
                      </FormLabel>
                      <Link
                        href="#"
                        className="text-muted-foreground hover:text-primary text-xs underline-offset-4 hover:underline"
                      >
                        {page_auth('forgot_your_password')}
                      </Link>
                    </div>
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
                {page_auth('signin')}
              </Button>

              <div className="text-muted-foreground text-center text-sm">
                {page_auth('do_not_have_an_account')}
                <Link
                  href={`/auth/signup?callbackUrl=${encodeURIComponent(redirectTo)}`}
                  className="text-primary font-medium underline-offset-4 hover:underline"
                >
                  {page_auth('signup')}
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
