'use client';

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
import { deleteDocument } from '@/features/document/client-api';
import type { Document } from '@/features/document/types';
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
    await deleteDocument(collection.id, document.id);
    toast.success(common_tips('delete_success'));
    setVisible(false);
    setTimeout(router.refresh, 300);
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
