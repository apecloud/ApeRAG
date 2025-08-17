'use client';

import { User } from '@/api';
import { createContext, useContext } from 'react';

import * as z from 'zod';

export const signInLocalSchema = z.object({
  username: z.string().min(1),
  password: z.string().min(1),
});

export const signUpLocalSchema = z.object({
  username: z.string().min(1),
  email: z.email(),
  password: z.string().min(1),
});

export type SignInOptions = {
  type?: 'local' | 'google' | 'github';
  data?: z.infer<typeof signInLocalSchema>;
  redirectTo: string;
};

export type SignUpOptions = {
  data: z.infer<typeof signUpLocalSchema>;
  redirectTo: string;
};

export type AppContextProps = {
  user?: User;
  signIn: (options?: SignInOptions) => void;
  signOut: () => void;
  signUp: (options: SignUpOptions) => void;
};

export const AppContext = createContext<AppContextProps>({
  user: undefined,
  signIn: () => {},
  signOut: () => {},
  signUp: () => {},
});

export const useAppContext = () => useContext(AppContext);
