'use client';
import {
  GlobalContext,
  signInLocalSchema,
  SignInOptions,
} from '@/hooks/use-global-context';
import { toast } from 'sonner';

import { User } from '@/api';
import { apiClient } from '@/lib/api/client';
import { useRouter } from 'next/navigation';
import { useCallback, useState } from 'react';

export const GlobalProvider = ({
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
      if (options?.type === undefined) {
        const callbackUrl = encodeURIComponent(options?.redirectTo || '/');
        router.push(`/auth/signin?callbackUrl=${callbackUrl}`);
        return;
      }

      if (options.type === 'local') {
        const { data } = signInLocalSchema.safeParse(options.data);
        if (!data) return;
        const res = await apiClient.defaultApi.loginPost({
          login: {
            username: data.username,
            password: data.password,
          },
        });
        if (res.status === 200) {
          setUser(res.data);
          router.push(`/workspace`);
        }
      }

      if (['github', 'google'].includes(options.type)) {
        try {
          const response = await fetch(
            `/api/v1/auth/${options.type}/authorize`,
          );
          const data = await response.json();

          console.log(data);

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

  const handleSignOut = useCallback(async () => {
    const res = await apiClient.defaultApi.logoutPost();
    if (res.status === 200) {
      setUser(undefined);
      setTimeout(router.refresh, 300);
    }
  }, [router]);

  return (
    <GlobalContext.Provider
      value={{
        user: _user,
        signIn: handleSignIn,
        signOut: handleSignOut,
      }}
    >
      {children}
    </GlobalContext.Provider>
  );
};
