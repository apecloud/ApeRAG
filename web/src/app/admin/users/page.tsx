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
          This module enables administrators to centrally control and organize
          all system users with secure access configurations. Efficiently manage
          identities, permissions, and authentication policies across your
          organization.
        </PageDescription>

        <UsersDataTable data={toJson(users)} />
      </PageContent>
    </PageContainer>
  );
}
