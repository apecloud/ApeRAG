'use client';

import { Department } from '@/api';
import { useBotContext } from '@/components/providers/bot-provider';
import {
  TreeMultipleSelect,
  TreeSelectItem,
} from '@/components/tree-multiple-select';

import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
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
import { cn } from '@/lib/utils';
import { Slot } from '@radix-ui/react-slot';
import { useTranslations } from 'next-intl';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { toast } from 'sonner';

export const BotSharing = ({ children }: { children?: React.ReactNode }) => {
  const [selectedDepartents, setSelectedDepartents] = useState<string[]>([]);
  const [departments, setDepartents] = useState<Department[]>([]);
  const { bot } = useBotContext();
  const page_bot = useTranslations('page_bot');
  const [sharingVisible, setSharingVisible] = useState<boolean>(false);

  const common_action = useTranslations('common.action');
  const common_tips = useTranslations('common.tips');

  const isSharedGlobal = useMemo(
    () => selectedDepartents.length === 1 && selectedDepartents[0] === '*',
    [selectedDepartents],
  );

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

        <Label
          data-checked={isSharedGlobal}
          className="hover:bg-accent/50 data-[checked=true]:bg-accent/50 flex cursor-pointer flex-row items-center justify-between gap-2 rounded-md border px-4 py-4 text-sm"
        >
          <div className="flex flex-col gap-1">
            <span>{page_bot('shared_global')}</span>
            <span className="text-muted-foreground text-xs">
              {page_bot('shared_global_tips')}
            </span>
          </div>

          <Checkbox
            checked={isSharedGlobal}
            onCheckedChange={(checked) => {
              setSelectedDepartents(checked ? ['*'] : []);
            }}
          />
        </Label>

        <div>
          <div className="mb-2 flex h-4 flex-row items-center justify-between gap-2 text-sm">
            <div>
              {page_bot('shared_department')} (
              {selectedDepartents.filter((d) => d !== '*').length})
            </div>
          </div>

          <div className="relative h-60">
            <TreeMultipleSelect
              options={departments as TreeSelectItem[]}
              values={selectedDepartents.filter((d) => d !== '*')}
              onValuesChange={(v) => setSelectedDepartents(v)}
              className={cn('relative h-60 overflow-auto rounded-md border')}
            />
            {isSharedGlobal && (
              <div className="bg-accent/50 absolute top-0 left-0 h-full w-full" />
            )}
          </div>
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
