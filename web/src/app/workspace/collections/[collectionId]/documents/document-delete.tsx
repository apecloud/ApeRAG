'use client';

import { Document } from '@/api';
import { useCollectionContext } from '@/components/providers/collection-provider';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog';
import { apiClient } from '@/lib/api/client';
import { Slot } from '@radix-ui/react-slot';
import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { toast } from 'sonner';

export const DocumentDelete = ({
  document,
  children,
}: {
  document: Document;
  children: React.ReactNode;
}) => {
  const { collection } = useCollectionContext();
  const [visible, setVisible] = useState<boolean>(false);
  const router = useRouter();

  const handleDelete = async () => {
    if (!collection.id || !document.id) return;
    const res =
      await apiClient.defaultApi.collectionsCollectionIdDocumentsDocumentIdDelete(
        {
          collectionId: collection.id,
          documentId: document.id,
        },
      );

    if (res.status === 200) {
      toast.success('Deleted successfully!');
      setVisible(false);
      setTimeout(router.refresh, 300);
    }
  };

  return (
    <AlertDialog open={visible} onOpenChange={() => setVisible(false)}>
      <AlertDialogTrigger asChild>
        <Slot
          onClick={(e) => {
            setVisible(true);
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
            This action cannot be undone. This will permanently delete file and
            remove your data from our servers.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel onClick={() => setVisible(false)}>
            Cancel
          </AlertDialogCancel>
          <AlertDialogAction onClick={() => handleDelete()}>
            Continue
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
};
