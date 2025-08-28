import {
  PageContainer,
  PageContent,
  PageHeader,
  PageTitle,
} from '@/components/page-container';
import { redirect } from 'next/navigation';

export default async function Page() {
  redirect('/admin/users');
  return (
    <PageContainer>
      <PageHeader />
      <PageContent>
        <PageTitle>Administrator</PageTitle>
      </PageContent>
    </PageContainer>
  );
}
