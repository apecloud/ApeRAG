import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';

export default async function Page() {
  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: 'Users' }]} />
      <PageContent>
        <PageTitle>Quota Management</PageTitle>
        <PageDescription>Manage user quotas and usage</PageDescription>
      </PageContent>
    </PageContainer>
  );
}
