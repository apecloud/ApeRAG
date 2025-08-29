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
import { useTranslations } from 'next-intl';
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
  const common_tips = useTranslations('common.tips');
  const common_action = useTranslations('common.action');
  const page_documents = useTranslations('page_documents');
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
      toast.success(common_tips('delete_success'));
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
          <AlertDialogTitle>{common_tips('confirm')}</AlertDialogTitle>
          <AlertDialogDescription>
            {page_documents('delete_document_confirm')}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel onClick={() => setVisible(false)}>
            {common_action('cancel')}
          </AlertDialogCancel>
          <AlertDialogAction onClick={() => handleDelete()}>
            {common_action('continue')}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
};
