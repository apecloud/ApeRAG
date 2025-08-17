'use client';
import {
  AppContext,
  signInLocalSchema,
  SignInOptions,
  SignUpOptions,
} from '@/hooks/use-app-context';
import { toast } from 'sonner';

import { User } from '@/api';
import { apiClient } from '@/lib/api/client';
import { useRouter } from 'next/navigation';
import { useCallback, useState } from 'react';

export const AppProvider = ({
  user,
  children,
}: {
  user?: User;
  children?: React.ReactNode;
}) => {
  const [_user, setUser] = useState<User | undefined>(user);

  const router = useRouter();
  const handleSignIn = useCallback(
    async (options?: SignInOptions) => {
      // redirect to sign in page
      if (options?.type === undefined) {
        const callbackUrl = encodeURIComponent(options?.redirectTo || '/');
        router.push(`/auth/signin?callbackUrl=${callbackUrl}`);
        return;
      }

      // signin with local credentials
      if (options.type === 'local') {
        const { data } = signInLocalSchema.safeParse(options.data);
        if (!data) return;

        try {
          const res = await apiClient.defaultApi.loginPost({
            login: data,
          });

          if (res.status === 200) {
            setUser(res.data);
            const callbackUrl = options.redirectTo || '/workspace';
            router.push(callbackUrl);
          }
          // eslint-disable-next-line @typescript-eslint/no-unused-vars
        } catch (err) {
          toast.error('Invalid credentials');
        }
      }

      // signin with third-party account
      if (['github', 'google'].includes(options.type)) {
        try {
          const response = await fetch(
            `/api/v1/auth/${options.type}/authorize`,
          );
          const data = await response.json();
          if (data.authorization_url) {
            window.location.href = data.authorization_url;
          }
        } catch (error) {
          console.error('OAuth error:', error);
          toast.error('authorize failed');
        }
      }
    },
    [router],
  );

  const handleSignUp = useCallback(
    async (params: SignUpOptions) => {
      try {
        const res = await apiClient.defaultApi.registerPost({
          register: params.data,
        });
        if (res.status === 200) {
          toast.success('Registration successful');
          router.push(
            `/auth/signin?callbackUrl=${encodeURIComponent(params.redirectTo || '/')}`,
          );
        }
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
      } catch (err) {
        toast.error('Invalid credentials');
      }
    },
    [router],
  );

  const handleSignOut = useCallback(async () => {
    const res = await apiClient.defaultApi.logoutPost();
    if (res.status === 200) {
      setUser(undefined);
      setTimeout(router.refresh, 300);
    }
  }, [router]);

  return (
    <AppContext.Provider
      value={{
        user: _user,
        signIn: handleSignIn,
        signOut: handleSignOut,
        signUp: handleSignUp,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};
