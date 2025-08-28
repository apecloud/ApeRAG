import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { CollectionHeader } from '../collection-header';
import { CollectionGraph } from './collection-graph';

export default async function Page() {
  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          {
            title: 'Collections',
            href: '/workspace/collections',
          },
          {
            title: 'Knowledge Graph',
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
