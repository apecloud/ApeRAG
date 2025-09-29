'use client';

import { Settings } from '@/api';
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
import { Switch } from '@/components/ui/switch';
import { apiClient } from '@/lib/api/client';
import { cn } from '@/lib/utils';
import { LaptopMinimalCheck, LoaderCircle } from 'lucide-react';
import { useTranslations } from 'next-intl';
import { useCallback, useEffect, useState } from 'react';
import { toast } from 'sonner';

const defaultValue = {
  use_mineru: false,
  mineru_api_token: '',
  use_doc_ray: false,
  use_markitdown: true,
};

export const ParserSettings = ({
  data: initData = defaultValue,
}: {
  data: Settings;
}) => {
  const [data, setData] = useState<Settings>({
    ...defaultValue,
    ...initData,
  });
  const admin_config = useTranslations('admin_config');
  const common_action = useTranslations('common.action');
  const common_tips = useTranslations('common.tips');
  const [checked, setChecked] = useState<boolean>(false);
  const [checking, setChecking] = useState<boolean>(false);

  const handleSave = useCallback(async () => {
    await apiClient.defaultApi.settingsPut({
      settings: data,
    });
    toast.success('Saved successfully');
  }, [data]);

  const handleSwitchChange = useCallback(
    async (key: keyof Settings, checked: boolean) => {
      const settings = { ...data, [key]: checked };
      setData(settings);
      await apiClient.defaultApi.settingsPut({
        settings,
      });
    },
    [data],
  );

  const handleCheckMineruToken = useCallback(async () => {
    if (!data.mineru_api_token) {
      toast.error(admin_config('mineru_api_token_required'));
      return;
    }

    setChecking(true);
    const res = await apiClient.defaultApi.settingsTestMineruTokenPost({
      settingsTestMineruTokenPostRequest: {
        token: data.mineru_api_token,
      },
    });
    if (res.data.status_code === 401) {
      toast.error(admin_config('mineru_api_token_invalid'));
    } else {
      setChecked(true);
      toast.success(common_tips('save_success'));
    }
    setChecking(false);
  }, [admin_config, common_tips, data.mineru_api_token]);

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
          <div className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>{admin_config('mineru_api')}</CardTitle>
              <CardDescription>
                {admin_config('mineru_api_description')}
              </CardDescription>
            </div>
            <Switch
              checked={data.use_mineru}
              onCheckedChange={(checked) =>
                handleSwitchChange('use_mineru', checked)
              }
            />
          </div>
        </CardHeader>

        <CardContent className={data.use_mineru ? 'block' : 'hidden'}>
          <div className="flex flex-row gap-4">
            <Input
              placeholder={admin_config('mineru_api_token')}
              value={data.mineru_api_token}
              onChange={(e) => {
                setData({ ...data, mineru_api_token: e.currentTarget.value });
              }}
            />
            <Button
              disabled={checking}
              variant="outline"
              onClick={handleCheckMineruToken}
            >
              {checking ? (
                <LoaderCircle className="animate-spin opacity-50" />
              ) : (
                <LaptopMinimalCheck />
              )}
              {admin_config('check')}
            </Button>
          </div>
          <div className="text-muted-foreground mt-2 text-sm">
            {admin_config('mineru_api_token_tips')}
          </div>
        </CardContent>

        <CardFooter
          className={cn('justify-end', data.use_mineru ? 'flex' : 'hidden')}
        >
          <Button disabled={!checked} onClick={handleSave}>
            {common_action('save')}
          </Button>
        </CardFooter>
      </Card>
      <Card>
        <CardHeader>
          <div className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>{admin_config('use_doc_ray')}</CardTitle>
              <CardDescription>
                {admin_config('use_doc_ray_description')}
              </CardDescription>
            </div>
            <Switch
              checked={data.use_doc_ray}
              onCheckedChange={(checked) =>
                handleSwitchChange('use_doc_ray', checked)
              }
            />
          </div>
        </CardHeader>
      </Card>
      <Card>
        <CardHeader>
          <div className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>{admin_config('use_markitdown')}</CardTitle>
              <CardDescription>
                {admin_config('use_markitdown_description')}
              </CardDescription>
            </div>
            <Switch
              checked={data.use_markitdown}
              onCheckedChange={(checked) =>
                handleSwitchChange('use_markitdown', checked)
              }
            />
          </div>
        </CardHeader>
      </Card>
    </>
  );
};
