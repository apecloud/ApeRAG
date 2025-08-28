import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';

import { CollectionForm } from '../../collection-form';
import { CollectionHeader } from '../collection-header';

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
            title: 'Settings',
          },
        ]}
      />
      <CollectionHeader />
      <PageContent>
        <CollectionForm action="edit" />
      </PageContent>
    </PageContainer>
  );
}
