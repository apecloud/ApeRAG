import { Markdown } from '@/components/markdown';
import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { DOCS_DIR } from '@/lib/docs';
import { getLocale } from '@/services/cookies';
import fs from 'fs';
import { getTranslations } from 'next-intl/server';

import { notFound } from 'next/navigation';
import path from 'path';

export default async function Page({
  params,
}: {
  params: Promise<{ group: string; paths: string[] }>;
}) {
  const page_docs = await getTranslations('page_docs');
  const { paths = [], group } = await params;
  const locale = await getLocale();
  const relativePath = path.join(group, ...paths);

  const localeMdxPath = path.join(DOCS_DIR, locale, `${relativePath}.mdx`);
  const localeMdPath = path.join(DOCS_DIR, locale, `${relativePath}.md`);

  let content;

  if (fs.existsSync(localeMdxPath)) {
    const mdxContent = fs.readFileSync(localeMdxPath, 'utf8');
    content = <Markdown>{mdxContent}</Markdown>;
  } else if (fs.existsSync(localeMdPath)) {
    const mdxContent = fs.readFileSync(localeMdPath, 'utf8');
    content = <Markdown>{mdxContent}</Markdown>;
  } else {
    notFound();
  }

  return (
    <PageContainer>
      <PageHeader breadcrumbs={[{ title: page_docs('metadata.title') }]} />
      <PageContent className="pt-12 pb-20">{content}</PageContent>
    </PageContainer>
  );
}
