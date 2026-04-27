'use client';

import {
  testMineruToken,
  updateSettings,
} from '@/features/admin/client-api';
import type { Settings } from '@/features/admin/types';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
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
import { cn } from '@/lib/utils';
import { LaptopMinimalCheck, LoaderCircle, RefreshCcw } from 'lucide-react';
import { useTranslations } from 'next-intl';
import { useCallback, useEffect, useState } from 'react';
import { toast } from 'sonner';

const defaultValue = {
  use_mineru: false,
  mineru_api_token: '',
  use_markitdown: true,
};

type HealthStatus = 'ok' | 'warning' | 'error' | 'disabled';
type SupportStatus = 'available' | 'limited' | 'unavailable' | 'disabled';

type ParserHealthItem = {
  key: string;
  label: string;
  status: HealthStatus;
  detail: string;
};

type ParserSupportTier = {
  key: string;
  label: string;
  category: 'official' | 'conditional' | 'enhanced' | 'optional';
  parser: string;
  formats: string[];
  status: SupportStatus;
  detail: string;
  requirements: string[];
};

type ParserHealthReport = {
  default_parser: string;
  parser_order: string[];
  available_extensions: string[];
  dependencies: ParserHealthItem[];
  services: ParserHealthItem[];
  support_tiers: ParserSupportTier[];
  warnings: string[];
  recommendations: string[];
};

const getHealthBadgeClassName = (status: HealthStatus | SupportStatus) => {
  switch (status) {
    case 'ok':
    case 'available':
      return 'border-emerald-200 bg-emerald-50 text-emerald-700';
    case 'warning':
    case 'limited':
      return 'border-amber-200 bg-amber-50 text-amber-700';
    case 'error':
    case 'unavailable':
      return 'border-red-200 bg-red-50 text-red-700';
    case 'disabled':
      return 'border-slate-200 bg-slate-50 text-slate-600';
  }
  return 'border-slate-200 bg-slate-50 text-slate-600';
};

const capitalize = (value: string) => value.charAt(0).toUpperCase() + value.slice(1);

const StatusBadge = ({ status }: { status: HealthStatus | SupportStatus }) => (
  <Badge variant="outline" className={getHealthBadgeClassName(status)}>
    {capitalize(status)}
  </Badge>
);

export const ParserSettings = ({
  data: initData = defaultValue,
}: {
  data?: Settings;
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
  const [health, setHealth] = useState<ParserHealthReport | null>(null);
  const [healthLoading, setHealthLoading] = useState<boolean>(true);

  const fetchParserHealth = useCallback(async () => {
    setHealthLoading(true);
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_BASE_PATH || ''}/api/v1/settings/parser_health`,
        {
          credentials: 'include',
          cache: 'no-store',
        },
      );
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const result = (await response.json()) as ParserHealthReport;
      setHealth(result);
    } catch (error) {
      toast.error(admin_config('parser_health_load_failed'));
      console.error(error);
    } finally {
      setHealthLoading(false);
    }
  }, [admin_config]);

  const handleSave = useCallback(async () => {
    await updateSettings(data);
    toast.success('Saved successfully');
    await fetchParserHealth();
  }, [data, fetchParserHealth]);

  const handleSwitchChange = useCallback(
    async (key: keyof Settings, checked: boolean) => {
      const settings = { ...data, [key]: checked };
      setData(settings);
      await updateSettings(settings);
      await fetchParserHealth();
    },
    [data, fetchParserHealth],
  );

  const handleCheckMineruToken = useCallback(async () => {
    if (!data.mineru_api_token) {
      toast.error(admin_config('mineru_api_token_required'));
      return;
    }

    setChecking(true);
    const res = await testMineruToken(data.mineru_api_token);
    if (res.status_code === 401) {
      toast.error(admin_config('mineru_api_token_invalid'));
    } else {
      setChecked(true);
      toast.success(common_tips('save_success'));
    }
    setChecking(false);
    await fetchParserHealth();
  }, [admin_config, common_tips, data.mineru_api_token, fetchParserHealth]);

  useEffect(() => {
    setData({
      ...defaultValue,
      ...initData,
    });
  }, [initData]);

  useEffect(() => {
    void fetchParserHealth();
  }, [fetchParserHealth]);

  return (
    <>
      <Card>
        <CardHeader>
          <div className="flex flex-row items-center justify-between gap-4">
            <div>
              <CardTitle>{admin_config('parser_health_title')}</CardTitle>
              <CardDescription>
                {admin_config('parser_health_description')}
              </CardDescription>
            </div>
            <Button
              variant="outline"
              disabled={healthLoading}
              onClick={() => void fetchParserHealth()}
            >
              <RefreshCcw className={cn(healthLoading ? 'animate-spin' : '')} />
              {admin_config('parser_health_refresh')}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {healthLoading && !health ? (
            <div className="text-muted-foreground flex items-center gap-2 text-sm">
              <LoaderCircle className="animate-spin" />
              {admin_config('parser_health_loading')}
            </div>
          ) : null}

          {health ? (
            <>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="rounded-lg border p-4">
                  <div className="mb-2 text-sm font-medium">
                    {admin_config('parser_health_default_parser')}
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="border-blue-200 bg-blue-50 text-blue-700">
                      {health.default_parser}
                    </Badge>
                  </div>
                </div>
                <div className="rounded-lg border p-4">
                  <div className="mb-2 text-sm font-medium">
                    {admin_config('parser_health_parser_order')}
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {health.parser_order.map((parserName) => (
                      <Badge key={parserName} variant="outline">
                        {parserName}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <div className="rounded-lg border p-4">
                  <div className="mb-3 text-sm font-medium">
                    {admin_config('parser_health_dependencies')}
                  </div>
                  <div className="space-y-3">
                    {health.dependencies.map((item) => (
                      <div key={item.key} className="space-y-1">
                        <div className="flex items-center justify-between gap-3">
                          <div className="text-sm font-medium">{item.label}</div>
                          <StatusBadge status={item.status} />
                        </div>
                        <div className="text-muted-foreground text-sm">
                          {item.detail}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="rounded-lg border p-4">
                  <div className="mb-3 text-sm font-medium">
                    {admin_config('parser_health_services')}
                  </div>
                  <div className="space-y-3">
                    {health.services.map((item) => (
                      <div key={item.key} className="space-y-1">
                        <div className="flex items-center justify-between gap-3">
                          <div className="text-sm font-medium">{item.label}</div>
                          <StatusBadge status={item.status} />
                        </div>
                        <div className="text-muted-foreground text-sm">
                          {item.detail}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="rounded-lg border p-4">
                <div className="mb-3 text-sm font-medium">
                  {admin_config('parser_health_support_matrix')}
                </div>
                <div className="space-y-4">
                  {health.support_tiers.map((tier) => (
                    <div key={tier.key} className="rounded-md border p-3">
                      <div className="mb-2 flex flex-wrap items-center justify-between gap-3">
                        <div className="flex flex-wrap items-center gap-2">
                          <div className="font-medium">{tier.label}</div>
                          <Badge variant="outline">{tier.category}</Badge>
                          <Badge variant="outline">{tier.parser}</Badge>
                        </div>
                        <StatusBadge status={tier.status} />
                      </div>
                      <div className="text-muted-foreground mb-2 text-sm">
                        {tier.detail}
                      </div>
                      <div className="mb-2 flex flex-wrap gap-2">
                        {tier.formats.map((format) => (
                          <Badge key={`${tier.key}-${format}`} variant="outline">
                            {format}
                          </Badge>
                        ))}
                      </div>
                      {tier.requirements.length > 0 ? (
                        <div className="text-xs text-slate-600">
                          {admin_config('parser_health_requirements')}:{' '}
                          {tier.requirements.join(', ')}
                        </div>
                      ) : null}
                    </div>
                  ))}
                </div>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <div className="rounded-lg border p-4">
                  <div className="mb-3 text-sm font-medium">
                    {admin_config('parser_health_warnings')}
                  </div>
                  {health.warnings.length > 0 ? (
                    <ul className="list-disc space-y-2 pl-5 text-sm">
                      {health.warnings.map((warning) => (
                        <li key={warning}>{warning}</li>
                      ))}
                    </ul>
                  ) : (
                    <div className="text-muted-foreground text-sm">
                      {admin_config('parser_health_none')}
                    </div>
                  )}
                </div>

                <div className="rounded-lg border p-4">
                  <div className="mb-3 text-sm font-medium">
                    {admin_config('parser_health_recommendations')}
                  </div>
                  {health.recommendations.length > 0 ? (
                    <ul className="list-disc space-y-2 pl-5 text-sm">
                      {health.recommendations.map((recommendation) => (
                        <li key={recommendation}>{recommendation}</li>
                      ))}
                    </ul>
                  ) : (
                    <div className="text-muted-foreground text-sm">
                      {admin_config('parser_health_none')}
                    </div>
                  )}
                </div>
              </div>
            </>
          ) : null}
        </CardContent>
      </Card>
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
              checked={data.use_mineru ?? undefined}
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
              value={data.mineru_api_token ?? ''}
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
              <CardTitle>{admin_config('use_markitdown')}</CardTitle>
              <CardDescription>
                {admin_config('use_markitdown_description')}
              </CardDescription>
            </div>
            <Switch
              checked={data.use_markitdown ?? undefined}
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
