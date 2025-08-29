import { SharedCollection } from '@/api';
import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { getServerApi } from '@/lib/api/server';
import { getTranslations } from 'next-intl/server';
import { CollectionList } from './collection-list';

export const dynamic = 'force-dynamic';

export default async function Page() {
  const serverApi = await getServerApi();
  const page_marketplace = await getTranslations('page_marketplace');
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
      <PageHeader
        breadcrumbs={[{ title: page_marketplace('metadata.title') }]}
      />
      <PageContent>
        <PageTitle>{page_marketplace('metadata.title')}</PageTitle>
        <PageDescription>
          {page_marketplace('metadata.description')}
        </PageDescription>

        <CollectionList collections={collections} />
      </PageContent>
    </PageContainer>
  );
}
