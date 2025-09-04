import {
  PageContainer,
  PageContent,
  PageDescription,
  PageTitle,
} from '@/components/page-container';

import { getTranslations } from 'next-intl/server';
export default async function Page() {
  const page_bot = await getTranslations('page_bot');

  return (
    <PageContainer>
      <PageContent>
        <PageTitle>{page_bot('metadata.title')}</PageTitle>
        <PageDescription>{page_bot('metadata.description')}</PageDescription>
      </PageContent>
    </PageContainer>
  );
}
