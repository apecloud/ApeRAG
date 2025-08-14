import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';

export default function Page() {
  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: 'Quotas' }]} />
      <PageContent>
        <PageTitle>Quotas</PageTitle>
        <PageDescription>Manage user quotas and usage</PageDescription>
      </PageContent>
    </PageContainer>
  );
}
