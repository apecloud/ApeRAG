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
import { apiClient } from '@/lib/api/client';
import { Slot } from '@radix-ui/react-slot';
import { useTranslations } from 'next-intl';
import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { toast } from 'sonner';

export const DocumentReBuildFailedIndex = ({
  children,
}: {
  children: React.ReactNode;
}) => {
  const { collection } = useCollectionContext();
  const common_tips = useTranslations('common.tips');
  const common_action = useTranslations('common.action');
  const page_documents = useTranslations('page_documents');
  const [visible, setVisible] = useState<boolean>(false);
  const router = useRouter();

  const handleRebuild = async () => {
    if (!collection.id) return;
    const res =
      await apiClient.defaultApi.collectionsCollectionIdRebuildFailedIndexesPost(
        {
          collectionId: collection.id,
        },
      );

    if (res.data.code === '200') {
      toast.success(page_documents('index_rebuild_failed_success'));
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
            {page_documents('index_rebuild_failed_confirm')}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel onClick={() => setVisible(false)}>
            {common_action('cancel')}
          </AlertDialogCancel>
          <AlertDialogAction onClick={() => handleRebuild()}>
            {common_action('continue')}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
};
