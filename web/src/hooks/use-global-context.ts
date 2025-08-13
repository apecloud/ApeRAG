'use client';

import { User } from '@/api';
import { createContext, useContext } from 'react';

export type GlobalContextProps = {
  user?: User;
};

export const GlobalContext = createContext<GlobalContextProps>({
  user: undefined,
});

export const useGlobalClientContext = () => useContext(GlobalContext);
