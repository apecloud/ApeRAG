'use client';

import { Card, CardContent } from '@/components/ui/card';
import type { QuotaInfo } from '@/features/quota/types';
import dynamic from 'next/dynamic';

const QuotaRadialChart = dynamic(
  () => import('./quota-radial-chart').then((r) => r.QuotaRadialChart),
  {
    ssr: false,
    loading: () => <QuotaChartSkeleton />,
  },
);

export const QuotaChartGrid = ({ quotas }: { quotas: QuotaInfo[] }) => {
  return (
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      {quotas.map((quota) => (
        <QuotaRadialChart key={quota.quota_type} data={quota} />
      ))}
    </div>
  );
};

const QuotaChartSkeleton = () => {
  return (
    <Card className="border-border/70 flex flex-col gap-0 overflow-hidden rounded-xl py-0">
      <CardContent className="space-y-4 p-6">
        <div className="bg-muted h-5 w-32 animate-pulse rounded" />
        <div className="bg-muted/80 mx-auto aspect-square max-h-[250px] w-full animate-pulse rounded-full" />
        <div className="flex justify-center gap-2">
          <div className="bg-muted h-6 w-20 animate-pulse rounded" />
          <div className="bg-muted h-6 w-20 animate-pulse rounded" />
        </div>
      </CardContent>
    </Card>
  );
};
