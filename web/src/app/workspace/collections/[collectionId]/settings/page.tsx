import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';

import { getTranslations } from 'next-intl/server';
import { CollectionForm } from '../../collection-form';
import { CollectionHeader } from '../collection-header';
import { CollectionVectorBackendCard } from './collection-vector-backend-card';

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
            title: page_collections('settings'),
          },
        ]}
      />
      <CollectionHeader />
      <PageContent className="pt-4">
        {/* task #61 P1-D3 (PR for #87): read-only deployment vector
            backend identity + capability matrix. Rendered above the
            edit form so the user sees the deployment-wide truth before
            tweaking per-collection knobs. */}
        <div className="mb-4">
          <CollectionVectorBackendCard />
        </div>
        <CollectionForm action="edit" />
      </PageContent>
    </PageContainer>
  );
}
