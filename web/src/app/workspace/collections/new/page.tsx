import {
  PageContainer,
  PageContent,
  PageDescription,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { CollectionForm } from '../collection-form';

export default async function Page() {
  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          { title: 'Collections', href: '/workspace/collections' },
          { title: 'New' },
        ]}
      />
      <PageContent>
        <PageTitle>New Collection</PageTitle>
        <PageDescription>
          By importing and systematically organizing your data sources into a
          structured dataset, you can significantly improve the contextual
          understanding and response accuracy of large language models (LLMs)
        </PageDescription>
        <CollectionForm action="add" />
      </PageContent>
    </PageContainer>
  );
}
