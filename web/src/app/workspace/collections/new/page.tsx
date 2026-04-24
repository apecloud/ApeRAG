import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { Database, Sparkles } from 'lucide-react';
import { Metadata } from 'next';
import { getTranslations } from 'next-intl/server';
import { CollectionForm } from '../collection-form';

export async function generateMetadata(): Promise<Metadata> {
  const page_collection_new = await getTranslations('page_collection_new');
  return {
    title: page_collection_new('metadata.title'),
    description: page_collection_new('metadata.description'),
  };
}

export default async function Page() {
  const page_collection_new = await getTranslations('page_collection_new');
  const page_collections = await getTranslations('page_collections');
  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          {
            title: page_collections('metadata.title'),
            href: '/workspace/collections',
          },
          { title: page_collection_new('metadata.title') },
        ]}
      />
      <PageContent className="max-w-5xl px-5 py-8 md:px-8 md:py-10">
        <div className="mb-8 grid gap-5 lg:grid-cols-[1fr_320px] lg:items-end">
          <div>
            <div className="text-muted-foreground font-mono text-[11px] tracking-[0.12em] uppercase">
              {page_collection_new('setup_label')}
            </div>
            <h1 className="font-serif mt-2 text-4xl leading-none font-normal tracking-normal md:text-[44px]">
              {page_collection_new('metadata.title')}
            </h1>
            <p className="text-muted-foreground mt-3 max-w-2xl text-sm leading-6">
              {page_collection_new('metadata.description')}
            </p>
          </div>
          <div className="border-border/70 bg-card flex items-start gap-3 rounded-xl border p-4 shadow-sm">
            <div className="bg-accent-soft text-accent-ink flex size-9 shrink-0 items-center justify-center rounded-lg">
              <Sparkles className="size-4" />
            </div>
            <div>
              <div className="text-sm font-medium">
                {page_collection_new('process_title')}
              </div>
              <div className="text-muted-foreground mt-1 text-xs leading-5">
                {page_collection_new('process_description')}
              </div>
            </div>
          </div>
        </div>
        <div className="border-border/70 bg-card mb-5 flex items-center gap-3 rounded-xl border px-4 py-3 shadow-sm">
          <div className="bg-muted text-foreground flex size-9 items-center justify-center rounded-lg">
            <Database className="size-4" />
          </div>
          <div className="text-muted-foreground text-sm">
            {page_collection_new('form_hint')}
          </div>
        </div>
        <CollectionForm action="add" />
      </PageContent>
    </PageContainer>
  );
}
