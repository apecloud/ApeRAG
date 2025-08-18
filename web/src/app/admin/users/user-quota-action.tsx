'use client';

import { QuotaUpdateRequest, User, UserQuotaInfo } from '@/api';
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
import { apiClient } from '@/lib/api/client';
import { zodResolver } from '@hookform/resolvers/zod';
import { Slot } from '@radix-ui/react-slot';
import _ from 'lodash';

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

export const UserQuotaAction = ({
  user,
  children,
}: {
  user: User;
  children?: React.ReactNode;
}) => {
  const [userQuotaInfo, setUserQuotaInfo] = useState<UserQuotaInfo>();
  const [visible, setVisible] = useState<boolean>(false);
  const quotaInfo = _.orderBy(userQuotaInfo?.quotas, ['quota_type'], ['desc']);

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
    const res = await apiClient.quotasApi.quotasGet({
      userId: user.id,
    });
    const data = res.data as UserQuotaInfo;

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

      const res = await apiClient.quotasApi.quotasUserIdPut({
        userId: user.id,
        quotaUpdateRequest: params,
      });
      if (res.data.success) {
        toast.success(res.data.message);
        setVisible(false);
      }
    },
    [user.id],
  );

  const handleRecalculate = useCallback(async () => {
    if (!user.id) return;
    const res = await apiClient.quotasApi.quotasUserIdRecalculatePost({
      userId: user.id,
    });
    if (res.data.success) {
      toast.success(res.data.message);
      getUserQuota();
    }
  }, [getUserQuota, user.id]);

  const content = useMemo(() => {
    if (_.isEmpty(quotaInfo)) {
      return (
        <>
          {_.times(4).map((index) => {
            return (
              <div key={index} className="flex w-full flex-col gap-2">
                <Skeleton className="h-[14px] w-1/2 rounded-md" />
                <Skeleton className="h-[36px] w-full rounded-md" />
              </div>
            );
          })}
        </>
      );
    } else {
      return quotaInfo?.map((info) => {
        const percent =
          info.quota_limit !== 0
            ? (info.current_usage * 100) / info.quota_limit
            : 0;
        return (
          <FormField
            key={info.quota_type}
            control={form.control}
            name={info.quota_type as keyof QuotaUpdateRequest}
            render={({ field }) => (
              <div>
                <FormItem>
                  <FormLabel>{_.startCase(info.quota_type)}</FormLabel>
                  <FormControl>
                    <Input
                      type="number"
                      {...field}
                      onChange={(e) => {
                        const v = Number(e.currentTarget.value);
                        field.onChange(v);
                      }}
                    />
                  </FormControl>
                </FormItem>
                <div className="my-1">
                  <Progress className="h-1" value={percent} />
                </div>
                <div className="text-muted-foreground flex flex-row justify-between text-sm">
                  <div>Usage: {info.current_usage}</div>
                  <div>{percent.toFixed(2)}%</div>
                </div>
              </div>
            )}
          />
        );
      });
    }
  }, [form.control, quotaInfo]);

  useEffect(() => {
    if (visible) {
      getUserQuota();
    }
  }, [getUserQuota, visible]);

  return (
    <Dialog open={visible} onOpenChange={() => setVisible(false)}>
      <DialogTrigger asChild>
        <Slot
          onClick={(e) => {
            setVisible(true);
            e.preventDefault();
          }}
        >
          {children}
        </Slot>
      </DialogTrigger>
      <DialogContent>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(handleUpdateQuota)}>
            <DialogHeader>
              <DialogTitle>User quota</DialogTitle>
              <DialogDescription asChild>
                <div className="flex flex-row gap-2">
                  {user.username && <div>{user.username}</div>}
                  {user.email && <div>{user.email}</div>}
                </div>
              </DialogDescription>
            </DialogHeader>

            <div className="flex flex-col gap-6 py-8">{content}</div>

            <DialogFooter className="flex flex-col sm:justify-between">
              <Button
                type="button"
                variant="secondary"
                onClick={() => handleRecalculate()}
                disabled={_.isEmpty(quotaInfo)}
              >
                Recalculate
              </Button>
              <div className="flex gap-2">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setVisible(false)}
                >
                  Cancel
                </Button>
                <Button type="submit" disabled={_.isEmpty(quotaInfo)}>
                  Save
                </Button>
              </div>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
};
