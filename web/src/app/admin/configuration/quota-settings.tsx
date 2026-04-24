'use client';

import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { updateSystemDefaultQuotas } from '@/features/admin/client-api';
import type { SystemDefaultQuotas } from '@/features/admin/types';
import { useTranslations } from 'next-intl';
import { useCallback, useEffect, useState } from 'react';
import { toast } from 'sonner';

const defaultValue = {
  use_mineru: false,
  mineru_api_token: '',
};

export const QuotaSettings = ({
  data: initData,
}: {
  data: SystemDefaultQuotas;
}) => {
  const [data, setData] = useState<SystemDefaultQuotas>({
    ...defaultValue,
    ...initData,
  });
  const admin_config = useTranslations('admin_config');
  const common_action = useTranslations('common.action');
  const page_quota = useTranslations('page_quota');
  const handleSave = useCallback(async () => {
    const res = await updateSystemDefaultQuotas({ quotas: data });
    if (res.success) {
      toast.success(res.message);
    }
  }, [data]);

  useEffect(() => {
    setData({
      ...defaultValue,
      ...initData,
    });
  }, [initData]);

  return (
    <>
      <Card className="border-border/70 gap-0 rounded-xl py-0 shadow-sm">
        <CardHeader className="border-border/70 border-b">
          <CardTitle className="font-serif text-2xl font-normal">
            {admin_config('system_default_quota')}
          </CardTitle>
          <CardDescription>
            {admin_config('system_default_quota_description')}
          </CardDescription>
        </CardHeader>

        <CardContent className="grid gap-4 p-5 md:grid-cols-2 xl:grid-cols-4">
          <div className="flex flex-col gap-2">
            <Label>{page_quota('bot_count.title')}</Label>
            <Input
              className="bg-background/70 rounded-xl"
              type="number"
              value={data.max_bot_count}
              onChange={(e) => {
                setData({
                  ...data,
                  max_bot_count: Number(e.currentTarget.value),
                });
              }}
            />
          </div>

          <div className="flex flex-col gap-2">
            <Label>{page_quota('collection_count.title')}</Label>
            <Input
              className="bg-background/70 rounded-xl"
              type="number"
              value={data.max_collection_count}
              onChange={(e) => {
                setData({
                  ...data,
                  max_collection_count: Number(e.currentTarget.value),
                });
              }}
            />
          </div>

          <div className="flex flex-col gap-2">
            <Label>{page_quota('document_count.title')}</Label>
            <Input
              className="bg-background/70 rounded-xl"
              type="number"
              value={data.max_document_count}
              onChange={(e) => {
                setData({
                  ...data,
                  max_document_count: Number(e.currentTarget.value),
                });
              }}
            />
          </div>

          <div className="flex flex-col gap-2">
            <Label>{page_quota('documents_per_collection.title')}</Label>
            <Input
              className="bg-background/70 rounded-xl"
              type="number"
              value={data.max_document_count_per_collection}
              onChange={(e) => {
                setData({
                  ...data,
                  max_document_count_per_collection: Number(
                    e.currentTarget.value,
                  ),
                });
              }}
            />
          </div>
        </CardContent>

        <CardFooter className="border-border/70 justify-end border-t p-4">
          <Button onClick={handleSave}>{common_action('save')}</Button>
        </CardFooter>
      </Card>
    </>
  );
};
