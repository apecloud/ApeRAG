'use client';

import { Button } from '@/components/ui/button';
import {
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
} from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { Skeleton } from '@/components/ui/skeleton';
import {
  getUserQuota as getUserQuotaApi,
  recalculateUserQuota,
  updateUserQuota,
} from '@/features/admin/client-api';
import type {
  QuotaUpdateRequest,
  QuotaUpdateResponse,
  UserQuotaInfo,
} from '@/features/admin/types';
import type { User } from '@/features/identity/types';
import { zodResolver } from '@hookform/resolvers/zod';
import { Gauge } from 'lucide-react';
import { useTranslations } from 'next-intl';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useForm } from 'react-hook-form';
import { toast } from 'sonner';
import * as z from 'zod';

const quotaSchema = z.object({
  max_collection_count: z.number().min(1),
  max_document_count: z.number().min(1),
  max_document_count_per_collection: z.number().min(1),
  max_bot_count: z.number().min(1),
});

const sortQuotaInfo = (quotaInfo?: UserQuotaInfo['quotas']) =>
  [...(quotaInfo ?? [])].sort((left, right) =>
    String(right.quota_type).localeCompare(String(left.quota_type)),
  );

export const UserQuotaDialogContent = ({
  user,
  onClose,
}: {
  user: User;
  onClose: () => void;
}) => {
  const [userQuotaInfo, setUserQuotaInfo] = useState<UserQuotaInfo>();
  const quotaInfo = useMemo(
    () => sortQuotaInfo(userQuotaInfo?.quotas),
    [userQuotaInfo?.quotas],
  );
  const hasQuotaInfo = quotaInfo.length > 0;
  const admin_users = useTranslations('admin_users');
  const page_quota = useTranslations('page_quota');
  const common_action = useTranslations('common.action');

  const form = useForm<z.infer<typeof quotaSchema>>({
    resolver: zodResolver(quotaSchema),
    defaultValues: {
      max_collection_count: 0,
      max_document_count: 0,
      max_document_count_per_collection: 0,
      max_bot_count: 0,
    },
  });

  const getUserQuota = useCallback(async () => {
    if (!user.id) return;
    const data = await getUserQuotaApi(user.id);

    data.quotas.forEach((quota) => {
      form.setValue(
        quota.quota_type as keyof QuotaUpdateRequest,
        quota.quota_limit,
      );
    });

    setUserQuotaInfo(data);
  }, [form, user.id]);

  const handleUpdateQuota = useCallback(
    async (values: z.infer<typeof quotaSchema>) => {
      const { data: params, error } = quotaSchema.safeParse(values);
      if (!user.id || error) return;

      const res = await updateUserQuota(user.id, params);
      if (res.success) {
        toast.success(res.message);
        onClose();
      }
    },
    [onClose, user.id],
  );

  const handleRecalculate = useCallback(async () => {
    if (!user.id) return;
    // `recalculateUserQuota` returns `unknown` (backend response not typed in
    // public OpenAPI; see features/admin/client-api.ts note). Narrow at the
    // call-site to read the success/message fields produced at runtime.
    const res = (await recalculateUserQuota(user.id)) as QuotaUpdateResponse;
    if (res?.success) {
      toast.success(res.message);
      getUserQuota();
    }
  }, [getUserQuota, user.id]);

  const content = useMemo(() => {
    if (!hasQuotaInfo) {
      return (
        <>
          {Array.from({ length: 4 }).map((_, index) => {
            return (
              <div key={index} className="flex w-full flex-col gap-2">
                <Skeleton className="h-[14px] w-1/2 rounded-md" />
                <Skeleton className="h-[36px] w-full rounded-md" />
              </div>
            );
          })}
        </>
      );
    }

    return quotaInfo.map((info) => {
      const percent =
        info.quota_limit !== 0
          ? (info.current_usage * 100) / info.quota_limit
          : 0;
      return (
        <FormField
          key={info.quota_type}
          control={form.control}
          name={info.quota_type as keyof QuotaUpdateRequest}
          render={({ field }) => {
            // @ts-expect-error i18n error
            const label = page_quota(info.quota_type);
            return (
              <div className="border-border/70 bg-muted rounded-xl border p-4">
                <FormItem>
                  <div className="mb-3 flex items-start justify-between gap-3">
                    <div>
                      <FormLabel className="font-medium">{label}</FormLabel>
                      <div className="text-muted-foreground mt-1 text-xs">
                        {page_quota('usage')}: {info.current_usage}
                      </div>
                    </div>
                    <div className="bg-accent-soft text-accent-ink flex size-9 shrink-0 items-center justify-center rounded-lg">
                      <Gauge className="size-4" />
                    </div>
                  </div>
                  <FormControl>
                    <Input
                      className="bg-background/70 rounded-xl font-mono"
                      type="number"
                      {...field}
                      onChange={(event) => {
                        const value = Number(event.currentTarget.value);
                        field.onChange(value);
                      }}
                    />
                  </FormControl>
                </FormItem>
                <div className="my-3">
                  <Progress className="bg-border h-1.5" value={percent} />
                </div>
                <div className="text-muted-foreground flex flex-row justify-between font-mono text-xs">
                  <div>
                    {page_quota('limit')}: {info.quota_limit}
                  </div>
                  <div>{percent.toFixed(2)}%</div>
                </div>
              </div>
            );
          }}
        />
      );
    });
  }, [form.control, hasQuotaInfo, page_quota, quotaInfo]);

  useEffect(() => {
    getUserQuota();
  }, [getUserQuota]);

  return (
    <DialogContent className="border-border/70 max-w-3xl rounded-xl p-0">
      <Form {...form}>
        <form onSubmit={form.handleSubmit(handleUpdateQuota)}>
          <DialogHeader className="border-border/70 border-b px-6 py-5">
            <DialogTitle className="font-serif text-2xl font-normal">
              {admin_users('user_quotas')}
            </DialogTitle>
            <DialogDescription asChild>
              <div className="text-muted-foreground mt-2 flex flex-wrap gap-2 text-sm">
                {user.username && <span>{user.username}</span>}
                {user.email && <span className="font-mono">{user.email}</span>}
              </div>
            </DialogDescription>
          </DialogHeader>

          <div className="grid gap-4 px-6 py-6 md:grid-cols-2">{content}</div>

          <DialogFooter className="border-border/70 flex flex-col border-t px-6 py-4 sm:flex-row sm:justify-between">
            <Button
              type="button"
              variant="outline"
              onClick={() => handleRecalculate()}
              disabled={!hasQuotaInfo}
            >
              {admin_users('user_quotas_recalculate')}
            </Button>
            <div className="flex gap-2">
              <Button type="button" variant="outline" onClick={onClose}>
                {common_action('cancel')}
              </Button>
              <Button type="submit" disabled={!hasQuotaInfo}>
                {common_action('save')}
              </Button>
            </div>
          </DialogFooter>
        </form>
      </Form>
    </DialogContent>
  );
};
