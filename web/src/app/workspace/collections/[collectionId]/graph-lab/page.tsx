import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { getTranslations } from 'next-intl/server';
import { CollectionHeader } from '../collection-header';
import { CollectionGraphLab } from './collection-graph-lab';

export default async function Page() {
  const pageCollections = await getTranslations('page_collections');
  const pageGraph = await getTranslations('page_graph');

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          {
            title: pageCollections('metadata.title'),
            href: '/workspace/collections',
          },
          {
            title: pageGraph('metadata.title'),
          },
          {
            title: 'Lab',
          },
        ]}
      />
      <div className="flex h-[calc(100vh-48px)] flex-col px-0">
        <CollectionHeader className="w-full" />
        <PageContent className="flex w-full max-w-[1440px] flex-1 flex-col">
          <CollectionGraphLab />
        </PageContent>
      </div>
    </PageContainer>
  );
}
