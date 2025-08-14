import { PageHeader } from '@/components/page-header';
import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';
import { DataTable } from './data-table';

export default async function Page() {
  const serverApi = await getServerApi();

  const res = await serverApi.defaultApi.apikeysGet();

  const data = res.data.items || [];

  return (
    <>
      <PageHeader
        title="API keys"
        description="The API key is your credential for accessing the system api. Please keep it safe.
"
        breadcrumbs={[{ title: 'API keys' }]}
      ></PageHeader>
      <div className="p-4">
        <DataTable data={toJson(data)} />
      </div>
    </>
  );
}
