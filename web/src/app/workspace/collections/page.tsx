import { CollectionView } from '@/api';
import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';

import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { getServerApi } from '@/lib/api/server';
import { toJson } from '@/lib/utils';
import { Plus } from 'lucide-react';
import { CollectionList } from './collection-list';

export default async function Page() {
  const serverApi = await getServerApi();

  let collections: CollectionView[] = [];
  try {
    const res = await serverApi.defaultApi.collectionsGet();
    collections = res.data.items || [];
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
  } catch (err) {}

  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: 'Collections' }]} />
      <PageContent>
        <PageTitle>Collections</PageTitle>
        <PageDescription>
          You can import and manage your data sources in dataset to enhance the
          context of LLM.
        </PageDescription>
        <Tabs defaultValue="creation" className="gap-4">
          <div className="flex flex-row items-center">
            <TabsList>
              <TabsTrigger value="creation">My Creations</TabsTrigger>
              <TabsTrigger value="subscribed">
                Subscribed Collections
              </TabsTrigger>
            </TabsList>
            <div className="ml-auto flex items-center gap-2">
              <Button>
                <Plus /> Add collection
              </Button>
            </div>
          </div>
          <TabsContent value="creation">
            <CollectionList
              collections={toJson(
                collections.filter((c) => !Boolean(c.subscription_id)),
              )}
            />
          </TabsContent>
          <TabsContent value="subscribed">
            <CollectionList
              collections={toJson(
                collections.filter((c) => Boolean(c.subscription_id)),
              )}
            />
          </TabsContent>
        </Tabs>
      </PageContent>
    </PageContainer>
  );
}
