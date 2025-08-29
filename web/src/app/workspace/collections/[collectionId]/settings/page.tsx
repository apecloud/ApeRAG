import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';

import { getTranslations } from 'next-intl/server';
import { CollectionForm } from '../../collection-form';
import { CollectionHeader } from '../collection-header';

export default async function Page() {
  const page_collections = await getTranslations('page_collections');
  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          {
            title: page_collections('metadata.title'),
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
