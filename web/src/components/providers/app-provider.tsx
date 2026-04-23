'use client';

import { toast } from 'sonner';

import {
  login as loginApi,
  logout as logoutApi,
  oauthAuthorize,
  register as registerApi,
} from '@/features/auth/client-api';
import type { User } from '@/features/identity/types';
import { useRouter } from 'next/navigation';
import { createContext, useCallback, useContext, useState } from 'react';

import * as z from 'zod';

type SignInOptions = {
  type?: 'local' | 'google' | 'github';
  data?: z.infer<typeof signInLocalSchema>;
  redirectTo: string;
};

type SignUpOptions = {
  data: z.infer<typeof signUpLocalSchema>;
  redirectTo: string;
};

type AppContextProps = {
  user?: User;
  signIn: (options?: SignInOptions) => void;
  signOut: () => void;
  signUp: (options: SignUpOptions) => void;
};

const AppContext = createContext<AppContextProps>({
  user: undefined,
  signIn: () => {},
  signOut: () => {},
  signUp: () => {},
});

export const signInLocalSchema = z.object({
  username: z.string().min(1),
  password: z.string().min(1),
});

export const signUpLocalSchema = z.object({
  username: z.string().min(1),
  email: z.email(),
  password: z.string().min(1),
});

export const useAppContext = () => useContext(AppContext);

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
          const user = await loginApi(data);
          setUser(user);
          const callbackUrl = options.redirectTo || '/workspace';
          router.push(callbackUrl);
          // eslint-disable-next-line @typescript-eslint/no-unused-vars
        } catch (err) {
          toast.error('Invalid credentials');
        }
      }

      // signin with third-party account
      if (options.type === 'github' || options.type === 'google') {
        try {
          const data = await oauthAuthorize(options.type);
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
        await registerApi(params.data);
        toast.success('Registration successful');
        router.push(
          `/auth/signin?callbackUrl=${encodeURIComponent(params.redirectTo || '/')}`,
        );
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
      } catch (err) {
        toast.error('Invalid credentials');
      }
    },
    [router],
  );

  const handleSignOut = useCallback(async () => {
    await logoutApi();
    setUser(undefined);
    setTimeout(router.refresh, 300);
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
