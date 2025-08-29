'use client';

import { SearchResult } from '@/api';
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

export const SearchDelete = ({
  searchResult,
  children,
}: {
  searchResult: SearchResult;
  children: React.ReactNode;
}) => {
  const { collection } = useCollectionContext();
  const [visible, setVisible] = useState<boolean>(false);
  const router = useRouter();
  const common_action = useTranslations('common.action');
  const common_tips = useTranslations('common.tips');
  const page_search = useTranslations('page_search');

  const handleDelete = async () => {
    if (!searchResult.id || !collection.id) return;
    const res =
      await apiClient.defaultApi.collectionsCollectionIdSearchesSearchIdDelete({
        collectionId: collection.id,
        searchId: searchResult.id,
      });

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
          <AlertDialogTitle>{common_tips('confirm')}</AlertDialogTitle>
          <AlertDialogDescription>
            {page_search('delete_confirm')}
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
