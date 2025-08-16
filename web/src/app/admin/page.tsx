import {
  PageContainer,
  PageContent,
  PageHeader,
  PageTitle,
} from '@/components/page-container';

export default async function Page() {
  return (
    <PageContainer>
      <PageHeader />
      <PageContent>
        <PageTitle>Administrator</PageTitle>
      </PageContent>
    </PageContainer>
  );
}
