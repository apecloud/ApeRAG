'use client';

import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { useTranslations } from 'next-intl';

export default function NotFoundPage() {
  const page_docs = useTranslations('page_docs');
  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: page_docs('metadata.title') }]} />
      <PageContent className="text-muted-foreground pt-50 text-center">
        page not found
      </PageContent>
    </PageContainer>
  );
}
