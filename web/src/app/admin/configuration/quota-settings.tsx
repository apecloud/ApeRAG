'use client';

import { SystemDefaultQuotas } from '@/api';
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
import { apiClient } from '@/lib/api/client';
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
    const res = await apiClient.quotasApi.systemDefaultQuotasPut({
      systemDefaultQuotasUpdateRequest: {
        quotas: data,
      },
    });
    if (res.data.success) {
      toast.success(res.data.message);
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
      <Card>
        <CardHeader>
          <CardTitle>{admin_config('system_default_quota')}</CardTitle>
          <CardDescription>
            {admin_config('system_default_quota_description')}
          </CardDescription>
        </CardHeader>

        <CardContent className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          <div className="flex flex-col gap-2">
            <Label>{page_quota('bot_count.title')}</Label>
            <Input
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

        <CardFooter className="justify-end">
          <Button onClick={handleSave}>{common_action('save')}</Button>
        </CardFooter>
      </Card>
    </>
  );
};
