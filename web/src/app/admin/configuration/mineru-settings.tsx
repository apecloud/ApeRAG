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
import { useCallback, useEffect, useState } from 'react';
import { toast } from 'sonner';

const defaultValue = {
  use_mineru: false,
  mineru_api_token: '',
};

export const MinerUSettings = ({
  data: initData = defaultValue,
}: {
  data: Settings;
}) => {
  const [data, setData] = useState<Settings>({
    ...defaultValue,
    ...initData,
  });

  const [checked, setChecked] = useState<boolean>(false);
  const [checking, setChecking] = useState<boolean>(false);

  const handleSave = useCallback(async () => {
    await apiClient.defaultApi.settingsPut({
      settings: data,
    });
    toast.success('Saved successfully');
  }, [data]);

  const handleUseMineruChange = useCallback(
    async (checked: boolean) => {
      const settings = { ...data, use_mineru: checked };
      setData(settings);
      await apiClient.defaultApi.settingsPut({
        settings,
      });
    },
    [data],
  );

  const handleCheckMineruToken = useCallback(async () => {
    if (!data.mineru_api_token) {
      toast.error('MinerU API Token is required.');
      return;
    }

    setChecking(true);
    const res = await apiClient.defaultApi.settingsTestMineruTokenPost({
      settingsTestMineruTokenPostRequest: {
        token: data.mineru_api_token,
      },
    });
    if (res.data.status_code === 401) {
      toast.error('Invalid token');
    } else {
      setChecked(true);
      toast.success('Successfully verified');
    }
    setChecking(false);
  }, [data.mineru_api_token]);

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
              <CardTitle>Use MinerU API</CardTitle>
              <CardDescription>
                When enabled, the system will prioritize using the MinerU API
                for document parsing to improve parsing quality and speed.
              </CardDescription>
            </div>
            <Switch
              checked={data.use_mineru}
              onCheckedChange={handleUseMineruChange}
            />
          </div>
        </CardHeader>

        <CardContent className={data.use_mineru ? 'block' : 'hidden'}>
          <div className="flex flex-row gap-4">
            <Input
              placeholder="MinerU API Token"
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
              Check
            </Button>
          </div>
          <div className="text-muted-foreground mt-2 text-sm">
            Tips: The official MinerU API Token is valid for 14 days, please
            replace it before it expires.
          </div>
        </CardContent>

        <CardFooter
          className={cn('justify-end', data.use_mineru ? 'flex' : 'hidden')}
        >
          <Button disabled={!checked} onClick={handleSave}>
            Save
          </Button>
        </CardFooter>
      </Card>
    </>
  );
};
