'use client';

import {
  GlobalContext,
  signInLocalSchema,
  SignInOptions,
} from '@/hooks/use-global-context';

import { User } from '@/api';
import { apiClient } from '@/lib/api/client';
import { useRouter } from 'next/navigation';
import { useState } from 'react';

export const GlobalProvider = ({
  user,
  children,
}: {
  user?: User;
  children?: React.ReactNode;
}) => {
  const [_user, setUser] = useState<User | undefined>(user);

  const router = useRouter();
  const handleSignIn = async (options?: SignInOptions) => {
    if (options === undefined) {
      const callbackUrl = encodeURIComponent(window.location.href);
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
        router.push(options.redirectTo);
      }
    }

    if (options.type === 'google') {
    }

    if (options.type === 'github') {
    }
  };

  const handleSignOut = async () => {
    const res = await apiClient.defaultApi.logoutPost();
    if (res.status === 200) {
      setUser(undefined);
      router.refresh();
    }
  };

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
