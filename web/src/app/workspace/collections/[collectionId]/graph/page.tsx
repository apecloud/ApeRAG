import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { getTranslations } from 'next-intl/server';
import { CollectionHeader } from '../collection-header';
import { CollectionGraph } from './collection-graph';

export default async function Page() {
  const page_collections = await getTranslations('page_collections');
  const page_graph = await getTranslations('page_graph');
  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          {
            title: page_collections('metadata.title'),
            href: '/workspace/collections',
          },
          {
            title: page_graph('metadata.title'),
          },
        ]}
      />
      <div className="flex h-[calc(100vh-48px)] flex-col px-0">
        <CollectionHeader className="w-full" />
        <PageContent className="flex w-full flex-1 flex-col">
          <CollectionGraph marketplace={false} />
        </PageContent>
      </div>
    </PageContainer>
  );
}
