import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';

export default async function Page() {
  // const [resSettings, resSystemDefaultQuotas] = await Promise.all([
  //   serverApi.defaultApi.settingsGet(),
  //   serverApi.quotasApi.systemDefaultQuotasGet(),
  // ]);

  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: 'Question Sets' }]} />
      <PageContent>
        <PageTitle>Question Sets</PageTitle>
        <PageDescription className="mb-8">
          An Evaluation Question Set is a focused collection of questions
          designed to systematically assess the effectiveness, impact.
        </PageDescription>

        <div className="flex flex-col gap-6"></div>
      </PageContent>
    </PageContainer>
  );
}
