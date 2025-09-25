'use client';

import { useBotContext } from '@/components/providers/bot-provider';
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
import { useTranslations } from 'next-intl';
import { useRouter } from 'next/navigation';
import { useCallback, useState } from 'react';

export const BotDelete = ({ children }: { children?: React.ReactNode }) => {
  const { bot } = useBotContext();
  const [deleteVisible, setDeleteVisible] = useState<boolean>(false);
  const router = useRouter();
  const common_action = useTranslations('common.action');
  const common_tips = useTranslations('common.tips');
  const page_bot = useTranslations('page_bot');

  const handleDelete = useCallback(async () => {
    if (bot?.id) {
      await apiClient.defaultApi.botsBotIdDelete({
        botId: bot.id,
      });
      setDeleteVisible(false);
      router.push('/bots');
    }
  }, [bot?.id, router]);

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
          <AlertDialogTitle>{common_tips('confirm')}</AlertDialogTitle>
          <AlertDialogDescription>
            {page_bot('delete_bot_confirm')}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogDescription></AlertDialogDescription>
        <AlertDialogFooter>
          <AlertDialogCancel onClick={() => setDeleteVisible(false)}>
            {common_action('cancel')}
          </AlertDialogCancel>
          <Button variant="destructive" onClick={() => handleDelete()}>
            {common_action('continue')}
          </Button>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
};
