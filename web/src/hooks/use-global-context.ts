"use client";

import { User } from "@/api";
import { createContext, useContext } from "react";

import * as z from "zod";

export const signInLocalSchema = z.object({
  username: z.string(),
  password: z.string(),
});

export type SignInOptions = {
  type: "local" | "google" | "github";
  data?: z.infer<typeof signInLocalSchema>;
  redirectTo: string;
};

export type GlobalContextProps = {
  user?: User;
  signIn: (options?: SignInOptions) => void;
  signOut: () => void;
};

export const GlobalContext = createContext<GlobalContextProps>({
  user: undefined,
  signIn: () => {},
  signOut: () => {},
});

export const useGlobalContext = () => useContext(GlobalContext);
