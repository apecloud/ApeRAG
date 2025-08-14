import { PageHeader } from '@/components/page-header';

export default function Page() {
  return (
    <>
      <PageHeader
        title="Audit Logs"
        description="View detailed audit records of system operations"
        breadcrumbs={[{ title: 'Audit Logs' }]}
      />
    </>
  );
}
