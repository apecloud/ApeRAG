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
        <PageTitle>Models & Providers</PageTitle>
        <PageDescription>
          This section allows administrators to manage and integrate third-party
          Large Language Model (LLM) providers and their respective models into
          the system. Configure API keys, model selection, rate limits, and
          other parameters to customize AI-powered functionalities.
        </PageDescription>
      </PageContent>
    </PageContainer>
  );
}
