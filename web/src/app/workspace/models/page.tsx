import { PageHeader } from '@/components/page-header';

export default function Page() {
  return (
    <>
      <PageHeader
        title="Models"
        description="Configure LLM providers and models"
        breadcrumbs={[{ title: 'Models' }]}
      />
    </>
  );
}
