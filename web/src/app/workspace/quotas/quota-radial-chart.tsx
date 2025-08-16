'use client';

import {
  Label,
  PolarGrid,
  PolarRadiusAxis,
  RadialBar,
  RadialBarChart,
} from 'recharts';

import { QuotaInfo } from '@/api';
import { Badge } from '@/components/ui/badge';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { ChartConfig, ChartContainer } from '@/components/ui/chart';
import { useMemo } from 'react';

export const QuotaRadialChart = ({ data }: { data: QuotaInfo }) => {
  const chartData = [{ usage: data.current_usage, fill: `var(--color-quota)` }];
  const percent =
    data.quota_limit === 0 ? 0 : (data.current_usage * 100) / data.quota_limit;
  const endAngle = (percent * 360) / 100;

  const chartConfig = {
    quota: {
      label: 'Bot Count',
      color: 'var(--primary)',
    },
  } satisfies ChartConfig;

  const quota = useMemo(() => {
    switch (data.quota_type) {
      case 'max_bot_count':
        return {
          title: 'Bot Count',
          description:
            'The upper limit for active bots in a system, preventing resource overuse and ensuring stability.',
        };
      case 'max_collection_count':
        return {
          title: 'Collection Count',
          description:
            'The maximum allowed collections in a database, controlling data structure complexity.',
        };
      case 'max_document_count':
        return {
          title: 'Documents Overall',
          description:
            'Caps the total documents across all collections, managing overall storage capacity.',
        };
      case 'max_document_count_per_collection':
        return {
          title: 'Documents per Collection',
          description:
            'Limits documents per collection, optimizing query performance and data balance.',
        };
      default:
        return { title: data.quota_type };
    }
  }, [data]);

  return (
    <Card className="flex flex-col gap-0 border-none">
      <CardHeader>
        <CardTitle>{quota.title}</CardTitle>
        <CardDescription>{quota.description}</CardDescription>
      </CardHeader>
      <CardContent className="flex-1 pb-0">
        <ChartContainer
          config={chartConfig}
          className="mx-auto aspect-square max-h-[250px]"
        >
          <RadialBarChart
            data={chartData}
            startAngle={0}
            endAngle={endAngle}
            innerRadius={80}
            outerRadius={110}
          >
            <PolarGrid
              gridType="circle"
              radialLines={false}
              stroke="none"
              className="first:fill-muted last:fill-background"
              polarRadius={[86, 74]}
            />
            <RadialBar dataKey="usage" background cornerRadius={10} />
            <PolarRadiusAxis tick={false} tickLine={false} axisLine={false}>
              <Label
                content={({ viewBox }) => {
                  if (viewBox && 'cx' in viewBox && 'cy' in viewBox) {
                    return (
                      <text
                        x={viewBox.cx}
                        y={viewBox.cy}
                        textAnchor="middle"
                        dominantBaseline="middle"
                      >
                        <tspan
                          x={viewBox.cx}
                          y={viewBox.cy}
                          className="fill-foreground text-4xl font-bold"
                        >
                          {`${percent.toFixed(2)}%`}
                        </tspan>
                        <tspan
                          x={viewBox.cx}
                          y={(viewBox.cy || 0) + 24}
                          className="fill-muted-foreground"
                        >
                          usage
                        </tspan>
                      </text>
                    );
                  }
                }}
              />
            </PolarRadiusAxis>
          </RadialBarChart>
        </ChartContainer>
      </CardContent>
      <CardFooter className="text-muted-foreground flex flex-row items-center justify-center gap-2 text-sm">
        <Badge variant="secondary">limit: {data.quota_limit}</Badge>
        <Badge variant="secondary">usage: {data.current_usage}</Badge>
      </CardFooter>
    </Card>
  );
};
