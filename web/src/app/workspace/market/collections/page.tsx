import { SharedCollection } from '@/api';
import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { CollectionList } from './collection-list';

export const dynamic = 'force-dynamic';

export default async function Page() {
  const serverApi = await getServerApi();

  let collections: SharedCollection[] = [];
  try {
    const res = await serverApi.defaultApi.marketplaceCollectionsGet({
      page: 1,
      pageSize: 100,
    });
    collections = res.data.items || [];
  } catch (err) {
    console.log(err);
  }

  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: 'Marketplace' }]} />
      <PageContent>
        <PageTitle>Marketplace</PageTitle>
        <PageDescription>
          Discover and subscribe to quality knowledge collections shared by
          community, expand your knowledge boundaries
        </PageDescription>

        <CollectionList collections={collections} />
      </PageContent>
    </PageContainer>
  );
}
