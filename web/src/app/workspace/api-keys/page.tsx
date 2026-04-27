import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';

import { listApiKeys } from '@/features/api-key/server-api';
import { toJson } from '@/lib/utils';
import { getTranslations } from 'next-intl/server';
import { ApiKeyTable } from './api-key-table';

export default async function Page() {
  const data = await listApiKeys();
  const page_api_keys = await getTranslations('page_api_keys');

  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: page_api_keys('metadata.title') }]} />
      <PageContent>
        <PageTitle>{page_api_keys('metadata.title')}</PageTitle>
        <PageDescription>
          {page_api_keys('metadata.description')}
        </PageDescription>
        <ApiKeyTable data={toJson(data)} />
      </PageContent>
    </PageContainer>
  );
}
