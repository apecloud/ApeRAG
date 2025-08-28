import { PageContainer, PageHeader } from '@/components/page-container';

export default function NotFoundPage() {
  return (
    <PageContainer>
      <PageHeader />
      <div className="mt-60 flex flex-col gap-4 text-center">
        <div className="text-5xl font-bold">404</div>
        <div className="text-4xl">Page Not Found</div>
      </div>
    </PageContainer>
  );
}
