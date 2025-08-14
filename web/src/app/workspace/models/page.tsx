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
      <PageHeader breadcrumbs={[{ title: 'Models' }]} />
      <PageContent>
        <PageTitle>Models</PageTitle>
        <PageDescription>Configure LLM providers and models</PageDescription>
      </PageContent>
    </PageContainer>
  );
}
