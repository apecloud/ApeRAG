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
} from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils';
import { zodResolver } from '@hookform/resolvers/zod';
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
      <Card className="bg-card/50">
        <CardContent>
          <div className="mb-8 text-center text-xl font-bold">Welcome back</div>
          {hasSocialLogin && (
            <div className="mb-4 grid gap-4">
              <div className="text-muted-foreground text-center text-sm">
                Login with a third-party account
              </div>
              {hasSocialGithubLogin && (
                <Button
                  variant="outline"
                  className="w-full"
                  onClick={() => signIn({ type: 'github', redirectTo })}
                >
                  <FaGithub />
                  Login with Github
                </Button>
              )}
              {hasSocialGoogleLogin && (
                <Button
                  variant="outline"
                  className="w-full"
                  onClick={() => signIn({ type: 'google', redirectTo })}
                >
                  <FaGoogle />
                  Login with Google
                </Button>
              )}

              <div className="after:border-border relative text-center text-sm after:absolute after:inset-0 after:top-1/2 after:z-0 after:flex after:items-center after:border-t">
                <span className="bg-card text-muted-foreground relative z-10 px-2">
                  Or continue with
                </span>
              </div>
            </div>
          )}
          <Form {...form}>
            <form
              onSubmit={form.handleSubmit(handleSignInLocal)}
              className="grid gap-6"
            >
              <FormField
                control={form.control}
                name="username"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Username</FormLabel>
                    <FormControl>
                      <Input {...field} placeholder="Username" />
                    </FormControl>
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="password"
                render={({ field }) => (
                  <FormItem>
                    <div className="flex justify-between">
                      <FormLabel>Password</FormLabel>
                      <Link
                        href="#"
                        className="text-muted-foreground hover:text-primary text-xs underline-offset-4 hover:underline"
                      >
                        Forgot your password?
                      </Link>
                    </div>
                    <FormControl>
                      <Input
                        type="password"
                        {...field}
                        placeholder="Password"
                      />
                    </FormControl>
                  </FormItem>
                )}
              />

              <Button type="submit" className="w-full">
                Sign In
              </Button>

              <div className="text-center text-sm">
                Don&apos;t have an account? &nbsp;
                <Link
                  href={`/auth/signup?callbaclUrl=${encodeURIComponent(redirectTo)}`}
                  className="underline underline-offset-4"
                >
                  Sign up
                </Link>
              </div>
            </form>
          </Form>
        </CardContent>
      </Card>
      <div className="text-muted-foreground *:[a]:hover:text-primary text-center text-xs text-balance *:[a]:underline *:[a]:underline-offset-4">
        By clicking continue, you agree to our{' '}
        <Link href="#">Terms of Service</Link> and{' '}
        <Link href="#">Privacy Policy</Link>.
      </div>
    </div>
  );
}
