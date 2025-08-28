import { CollectionView } from '@/api';
import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';

import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';
import { CollectionList } from './collection-list';

export const dynamic = 'force-dynamic';

export default async function Page() {
  const serverApi = await getServerApi();

  let collections: CollectionView[] = [];
  try {
    const res = await serverApi.defaultApi.collectionsGet({
      page: 1,
      pageSize: 100,
      includeSubscribed: true,
    });
    collections = res.data.items || [];
  } catch (err) {
    console.log(err);
  }

  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: 'Collections' }]} />
      <PageContent>
        <PageTitle>Collections</PageTitle>
        <PageDescription>
          By importing and systematically organizing your data sources into a
          structured dataset, you can significantly improve the contextual
          understanding and response accuracy of large language models (LLMs)
        </PageDescription>
        <CollectionList collections={toJson(collections)} />
      </PageContent>
    </PageContainer>
  );
}
