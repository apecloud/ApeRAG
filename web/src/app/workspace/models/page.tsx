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
        <PageDescription>
          This section allows you to connect and customize your preferred Large
          Language Model (LLM) providers and models for personal use. Set up API
          keys, choose models, and adjust settings to enhance your AI
          experience.
        </PageDescription>
      </PageContent>
    </PageContainer>
  );
}
