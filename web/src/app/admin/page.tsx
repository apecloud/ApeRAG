import {
  PageContainer,
  PageContent,
  PageHeader,
  PageTitle,
} from '@/components/page-container';

export default async function Page() {
  return (
    <PageContainer>
      <PageHeader defaultBreadcrumb={{ title: 'Admin', href: '/admin' }} />
      <PageContent>
        <PageTitle>Administrator</PageTitle>
      </PageContent>
    </PageContainer>
  );
}
