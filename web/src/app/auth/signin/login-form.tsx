'use client';

import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  signInLocalSchema,
  useGlobalContext,
} from '@/hooks/use-global-context';
import { cn } from '@/lib/utils';
import { useSearchParams } from 'next/navigation';
import { FormEvent, useCallback, useMemo } from 'react';
import { FaGithub, FaGoogle } from 'react-icons/fa6';

export function LoginForm({
  className,
  methods,
  ...props
}: React.ComponentProps<'div'> & {
  methods: string[];
}) {
  const searchParams = useSearchParams();
  const redirectTo = searchParams.get('callbackUrl') || '/';
  const { signIn } = useGlobalContext();

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
    async (e: FormEvent<HTMLFormElement>) => {
      e.preventDefault();
      const formData = new FormData(e.currentTarget);
      const payload = Object.fromEntries(formData.entries());
      const { data, error } = signInLocalSchema.safeParse(payload);

      if (!error) {
        await signIn({
          type: 'local',
          data,
          redirectTo,
        });
      }
    },
    [redirectTo, signIn],
  );

  return (
    <div className={cn('flex flex-col gap-6', className)} {...props}>
      <Card>
        <CardContent>
          <div className="mb-4 text-center text-xl font-bold">Welcome back</div>
          {hasSocialLogin && (
            <>
              <div className="text-muted-foreground mb-6 text-center text-sm">
                Login with your Github or Google account
              </div>
              <div className="flex flex-col gap-4 text-sm">
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
              </div>
              <div className="after:border-border relative my-6 text-center text-sm after:absolute after:inset-0 after:top-1/2 after:z-0 after:flex after:items-center after:border-t">
                <span className="bg-card text-muted-foreground relative z-10 px-2">
                  Or continue with
                </span>
              </div>
            </>
          )}

          <form onSubmit={handleSignInLocal}>
            <div className="grid gap-6">
              <div className="grid gap-6">
                <div className="grid gap-3">
                  <Label htmlFor="username">Username</Label>
                  <Input id="username" name="username" required />
                </div>
                <div className="grid gap-3">
                  <div className="flex items-center">
                    <Label htmlFor="password">Password</Label>
                    <a
                      href="#"
                      className="ml-auto text-sm underline-offset-4 hover:underline"
                    >
                      Forgot your password?
                    </a>
                  </div>
                  <Input
                    id="password"
                    name="password"
                    type="password"
                    required
                  />
                </div>
                <Button type="submit" className="w-full">
                  Login
                </Button>
              </div>
              <div className="text-center text-sm">
                Don&apos;t have an account?{' '}
                <a href="#" className="underline underline-offset-4">
                  Sign up
                </a>
              </div>
            </div>
          </form>
        </CardContent>
      </Card>
      {/* <div className="text-muted-foreground *:[a]:hover:text-primary text-center text-xs text-balance *:[a]:underline *:[a]:underline-offset-4">
        By clicking continue, you agree to our <a href="#">Terms of Service</a>{' '}
        and <a href="#">Privacy Policy</a>.
      </div> */}
    </div>
  );
}
