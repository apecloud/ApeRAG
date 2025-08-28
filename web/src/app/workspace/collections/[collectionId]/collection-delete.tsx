'use client';

import { useCollectionContext } from '@/components/providers/collection-provider';
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog';
import { Button } from '@/components/ui/button';
import { apiClient } from '@/lib/api/client';
import { Slot } from '@radix-ui/react-slot';
import { useRouter } from 'next/navigation';
import { useCallback, useState } from 'react';

export const CollectionDelete = ({
  children,
}: {
  children?: React.ReactNode;
}) => {
  const { collection } = useCollectionContext();
  const [deleteVisible, setDeleteVisible] = useState<boolean>(false);
  const router = useRouter();

  const handleDelete = useCallback(async () => {
    if (collection?.id) {
      const res = await apiClient.defaultApi.collectionsCollectionIdDelete({
        collectionId: collection.id,
      });
      if (res?.status === 200) {
        setDeleteVisible(false);
        router.push('/workspace/collections');
      }
    }
  }, [collection?.id, router]);

  return (
    <AlertDialog
      open={deleteVisible}
      onOpenChange={() => setDeleteVisible(false)}
    >
      <AlertDialogTrigger asChild>
        <Slot
          onClick={(e) => {
            setDeleteVisible(true);
            e.preventDefault();
          }}
        >
          {children}
        </Slot>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
          <AlertDialogDescription>
            This action cannot be undone. This will permanently delete
            collection and remove your documents from our servers.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogDescription></AlertDialogDescription>
        <AlertDialogFooter>
          <AlertDialogCancel onClick={() => setDeleteVisible(false)}>
            Cancel
          </AlertDialogCancel>
          <Button variant="destructive" onClick={() => handleDelete()}>
            Continue
          </Button>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
};
