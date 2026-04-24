import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { Card, CardContent } from '@/components/ui/card';
import { listUsers } from '@/features/admin/server-api';
import { toJson } from '@/lib/utils';
import { ShieldCheck, UserCheck, UsersRound } from 'lucide-react';
import { Metadata } from 'next';
import { getTranslations } from 'next-intl/server';
import type { ReactNode } from 'react';
import { UsersDataTable } from './users-data-table';

export async function generateMetadata(): Promise<Metadata> {
  const admin_users = await getTranslations('admin_users');
  return {
    title: admin_users('metadata.title'),
    description: admin_users('metadata.description'),
  };
}

export default async function Page() {
  const admin_users = await getTranslations('admin_users');
  const res = await listUsers();

  const users = res.items || [];
  const activeUsers = users.filter((user) => user.is_active).length;
  const adminUsers = users.filter((user) => user.role === 'admin').length;

  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: admin_users('metadata.title') }]} />
      <PageContent className="max-w-7xl px-5 py-8 md:px-8 md:py-10">
        <div className="mb-8 flex flex-col gap-5 lg:flex-row lg:items-end">
          <div className="min-w-0 flex-1">
            <div className="text-muted-foreground font-mono text-[11px] tracking-[0.12em] uppercase">
              {admin_users('metadata.label')}
            </div>
            <h1 className="mt-2 font-serif text-4xl leading-none font-normal tracking-normal md:text-[44px]">
              {admin_users('metadata.title')}
            </h1>
            <p className="text-muted-foreground mt-3 max-w-2xl text-sm leading-6">
              {admin_users('metadata.description')}
            </p>
          </div>
        </div>
        <div className="mb-6 grid gap-3 md:grid-cols-3">
          <AdminMetric
            icon={<UsersRound className="size-4" />}
            label={admin_users('metric_users')}
            value={users.length}
          />
          <AdminMetric
            icon={<UserCheck className="size-4" />}
            label={admin_users('metric_active')}
            value={activeUsers}
          />
          <AdminMetric
            icon={<ShieldCheck className="size-4" />}
            label={admin_users('metric_admins')}
            value={adminUsers}
          />
        </div>
        <UsersDataTable data={toJson(users)} />
      </PageContent>
    </PageContainer>
  );
}

const AdminMetric = ({
  icon,
  label,
  value,
}: {
  icon: ReactNode;
  label: string;
  value: number;
}) => (
  <Card className="border-border/70 gap-0 rounded-xl py-0">
    <CardContent className="flex items-center gap-3 p-4">
      <div className="bg-accent-soft text-accent-ink flex size-9 items-center justify-center rounded-lg">
        {icon}
      </div>
      <div>
        <div className="font-mono text-xl leading-none tabular-nums">
          {value}
        </div>
        <div className="text-muted-foreground mt-1 text-xs">{label}</div>
      </div>
    </CardContent>
  </Card>
);
