'use client';

import { GlobalContext, GlobalContextProps } from '@/hooks/use-global-context';

export const GlobalProvider = ({
  user,

  children,
}: GlobalContextProps & { children: React.ReactNode }) => {
  return (
    <GlobalContext.Provider
      value={{
        user,
      }}
    >
      {children}
    </GlobalContext.Provider>
  );
};
