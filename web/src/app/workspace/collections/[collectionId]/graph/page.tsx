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
      <CollectionHeader />
      <PageContent>
        <CollectionGraph marketplace={false} />
      </PageContent>
    </PageContainer>
  );
}
