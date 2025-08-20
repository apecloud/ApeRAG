import { PageHeader } from '@/components/page-header';

export default function Page() {
  return (
    <>
      <PageHeader
        title="Quotas"
        description="Manage user quotas and usage"
        breadcrumbs={[{ title: 'Quotas' }]}
      />
    </>
  );
}
