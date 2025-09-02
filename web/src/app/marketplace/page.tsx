import { SharedCollection } from '@/api';
import {
  PageContainer,
  PageContent,
  PageDescription,
  PageTitle,
} from '@/components/page-container';
import { Button } from '@/components/ui/button';
import { getServerApi } from '@/lib/api/server';
import { BookOpen } from 'lucide-react';
import { getTranslations } from 'next-intl/server';
import Link from 'next/link';
import { CollectionList } from './collection-list';

export const dynamic = 'force-dynamic';

export default async function Page() {
  const serverApi = await getServerApi();
  const page_marketplace = await getTranslations('page_marketplace');
  const sidebar_workspace = await getTranslations('sidebar_workspace');
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
      <PageContent>
        <div className="flex">
          <PageTitle>{page_marketplace('metadata.title')}</PageTitle>
          <Button className="ml-auto" asChild>
            <Link href="/workspace/collections">
              <BookOpen /> {sidebar_workspace('collections')}
            </Link>
          </Button>
        </div>
        <PageDescription>
          {page_marketplace('metadata.description')}
        </PageDescription>

        <CollectionList collections={collections} />
      </PageContent>
    </PageContainer>
  );
}
