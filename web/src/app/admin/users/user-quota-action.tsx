'use client';

import { Dialog, DialogContent, DialogTrigger } from '@/components/ui/dialog';
import { Skeleton } from '@/components/ui/skeleton';
import type { User } from '@/features/identity/types';
import { Slot } from '@radix-ui/react-slot';
import dynamic from 'next/dynamic';
import type { ReactNode } from 'react';
import { useState } from 'react';

const UserQuotaDialogContent = dynamic(
  () =>
    import('./user-quota-dialog-content').then((mod) => ({
      default: mod.UserQuotaDialogContent,
    })),
  {
    ssr: false,
    loading: () => (
      <DialogContent className="border-border/70 max-w-3xl rounded-xl p-6">
        <div className="grid gap-4 md:grid-cols-2">
          {Array.from({ length: 4 }).map((_, index) => (
            <div key={index} className="flex w-full flex-col gap-2">
              <Skeleton className="h-[14px] w-1/2 rounded-md" />
              <Skeleton className="h-[36px] w-full rounded-md" />
            </div>
          ))}
        </div>
      </DialogContent>
    ),
  },
);

export const UserQuotaAction = ({
  user,
  children,
}: {
  user: User;
  children?: ReactNode;
}) => {
  const [visible, setVisible] = useState(false);

  return (
    <Dialog open={visible} onOpenChange={setVisible}>
      <DialogTrigger asChild>
        <Slot
          onClick={(event) => {
            setVisible(true);
            event.preventDefault();
          }}
        >
          {children}
        </Slot>
      </DialogTrigger>
      {visible ? (
        <UserQuotaDialogContent user={user} onClose={() => setVisible(false)} />
      ) : null}
    </Dialog>
  );
};
