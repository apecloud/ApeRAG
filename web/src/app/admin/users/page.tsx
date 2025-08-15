import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';
import { UsersDataTable } from './users-data-table';

export default async function Page() {
  const apiServer = await getServerApi();
  const res = await apiServer.defaultApi.usersGet();

  const users = res.data.items || [];

  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: 'Users' }]} />
      <PageContent>
        <PageTitle>User Management</PageTitle>
        <PageDescription>
          Manage user identities including password resets, creating and
          provisioning, blocking and deleting users.
        </PageDescription>

        <UsersDataTable data={toJson(users)} />
      </PageContent>
    </PageContainer>
  );
}
