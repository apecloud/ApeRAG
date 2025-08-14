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
      <PageHeader breadcrumbs={[{ title: 'Audit Logs' }]} />
      <PageContent>
        <PageTitle>Audit Logs</PageTitle>
        <PageDescription>
          View detailed audit records of system operations
        </PageDescription>
      </PageContent>
    </PageContainer>
  );
}
