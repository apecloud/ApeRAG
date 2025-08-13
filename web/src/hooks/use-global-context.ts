'use client';

import { createContext, useContext } from 'react';

export type GlobalContextProps = {
  user: UserEntity | null;
};

export const GlobalContext = createContext<GlobalContextProps>({
  user: null,
});

export const useGlobalClientContext = () => useContext(GlobalContext);
