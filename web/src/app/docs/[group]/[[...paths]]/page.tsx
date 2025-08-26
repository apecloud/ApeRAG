import { Markdown } from '@/components/markdown';
import { DOCS_DIR } from '@/lib/docs';
import fs from 'fs';

import { notFound } from 'next/navigation';
import path from 'path';

export default async function Page({
  params,
}: {
  params: Promise<{ group: string; paths: string[] }>;
}) {
  const { paths = [], group } = await params;

  const relativePath = path.join(group, ...paths);
  const mdxPath = path.join(DOCS_DIR, `${relativePath}.mdx`);

  if (fs.existsSync(mdxPath)) {
    const mdxContent = fs.readFileSync(mdxPath, 'utf8');
    return <Markdown>{mdxContent}</Markdown>;
  } else {
    notFound();
  }
}
