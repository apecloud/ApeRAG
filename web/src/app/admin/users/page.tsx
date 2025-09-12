import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';
import { Metadata } from 'next';
import { getTranslations } from 'next-intl/server';
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
  const apiServer = await getServerApi();
  const res = await apiServer.defaultApi.usersGet();

  const users = res.data.items || [];

  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: admin_users('metadata.title') }]} />
      <PageContent>
        <PageTitle>{admin_users('metadata.title')}</PageTitle>
        <PageDescription>{admin_users('metadata.description')}</PageDescription>
        <UsersDataTable data={toJson(users)} />
      </PageContent>
    </PageContainer>
  );
}
