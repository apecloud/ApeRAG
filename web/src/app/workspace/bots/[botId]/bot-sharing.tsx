'use client';

import { Department } from '@/api';
import { useBotContext } from '@/components/providers/bot-provider';
import {
  TreeMultipleSelect,
  TreeSelectItem,
} from '@/components/tree-multiple-select';

import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { apiClient } from '@/lib/api/client';
import { Slot } from '@radix-ui/react-slot';
import { useTranslations } from 'next-intl';
import { useCallback, useEffect, useState } from 'react';
import { toast } from 'sonner';

export const BotSharing = ({ children }: { children?: React.ReactNode }) => {
  const [selectedDepartents, setSelectedDepartents] = useState<string[]>([]);
  const [departments, setDepartents] = useState<Department[]>([]);
  const { bot } = useBotContext();
  const page_bot = useTranslations('page_bot');
  const [sharingVisible, setSharingVisible] = useState<boolean>(false);

  const common_action = useTranslations('common.action');
  const common_tips = useTranslations('common.tips');

  const handleSharing = useCallback(async () => {
    if (!bot?.id) return;
    await apiClient.defaultApi.botsBotIdSharingPost({
      botId: bot.id,
      botPublishRequest: {
        group_ids: selectedDepartents,
      },
    });
    setSharingVisible(false);
    toast.success(common_tips('update_success'));
  }, [bot?.id, common_tips, selectedDepartents]);

  const loadData = useCallback(async () => {
    if (!bot?.id) return;
    const res = await apiClient.defaultApi.departmentsGet();
    setDepartents(res.data.items || []);
  }, [bot?.id]);

  const loadShare = useCallback(async () => {
    if (!bot?.id) return;
    const res = await apiClient.defaultApi.botsBotIdSharingGet({
      botId: bot.id,
    });
    // @ts-expect-error api define error
    setSelectedDepartents(res.data.group_ids || []);
  }, [bot?.id]);

  useEffect(() => {
    loadShare();
  }, [loadShare]);

  useEffect(() => {
    if (sharingVisible) {
      loadData();
      loadShare();
    }
  }, [departments.length, loadData, loadShare, sharingVisible]);

  return (
    <Dialog open={sharingVisible} onOpenChange={() => setSharingVisible(false)}>
      <DialogTrigger asChild>
        <Slot
          onClick={(e) => {
            setSharingVisible(true);
            e.preventDefault();
          }}
        >
          {children}
        </Slot>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{page_bot('share_settings')}</DialogTitle>
          <DialogDescription>
            {page_bot('share_settings_tips')}
          </DialogDescription>
        </DialogHeader>
        <div>
          <div className="mb-2 flex h-4 flex-row items-center gap-2 text-sm">
            <Label>
              {page_bot('shared_department')} ({selectedDepartents.length})
            </Label>
          </div>
          <TreeMultipleSelect
            options={departments as TreeSelectItem[]}
            className="h-60 overflow-auto rounded-md border"
            values={selectedDepartents}
            onValuesChange={(v) => setSelectedDepartents(v)}
          />
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => setSharingVisible(false)}>
            {common_action('cancel')}
          </Button>
          <Button onClick={() => handleSharing()}>
            {common_action('continue')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
