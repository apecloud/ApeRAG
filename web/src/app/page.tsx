import { AppTopbar } from '@/components/app-topbar';
import { PageContainer, PageContent } from '@/components/page-container';
import { Button } from '@/components/ui/button';
import { ArrowRight, Mail } from 'lucide-react';
import Link from 'next/link';

const features = [
  {
    title: 'Hybrid Retrieval Engine',
    description:
      'Combines Graph RAG, vector search, and full-text search for comprehensive document understanding and retrieval.',
  },
  {
    title: 'Graph RAG with LightRAG',
    description:
      'Enhanced version of LightRAG for advanced graph-based knowledge extraction, enabling deep relational and contextual queries.',
  },
  {
    title: 'MinerU Integration',
    description:
      'Advanced document parsing service providing superior parsing for complex documents, tables, formulas, and scientific content with optional GPU acceleration.',
  },
  {
    title: 'Production-Grade Deployment',
    description:
      'Full Kubernetes support with Helm charts and KubeBlocks integration for simplified deployment of production-grade databases.',
  },
  {
    title: 'Multimodal Processing',
    description:
      'Supports various document formats (PDF, DOCX, etc.) with intelligent content extraction and structure recognition.',
  },
  {
    title: 'Enterprise Management',
    description:
      'Built-in audit logging, LLM model management, graph visualization, and comprehensive document management interface.',
  },
];

export default function Home() {
  return (
    <>
      <AppTopbar />
      <PageContainer className="relative px-6">
        <div
          aria-hidden="true"
          className="absolute inset-x-0 -top-40 -z-10 transform-gpu overflow-hidden blur-3xl sm:-top-80"
        >
          <div
            style={{
              clipPath:
                'polygon(74.1% 44.1%, 100% 61.6%, 97.5% 26.9%, 85.5% 0.1%, 80.7% 2%, 72.5% 32.5%, 60.2% 62.4%, 52.4% 68.1%, 47.5% 58.3%, 45.2% 34.5%, 27.5% 76.7%, 0.1% 64.9%, 17.9% 100%, 27.6% 76.8%, 76.1% 97.7%, 74.1% 44.1%)',
            }}
            className="relative left-[calc(50%-11rem)] aspect-1155/678 w-144.5 -translate-x-1/2 rotate-30 bg-linear-to-tr from-[#ff80b5] to-[#9089fc] opacity-30 sm:left-[calc(50%-30rem)] sm:w-288.75"
          ></div>
        </div>
        <PageContent className="mx-auto max-w-300 py-48">
          <div className="hidden sm:mb-8 sm:flex sm:justify-center">
            <div className="flex flex-row items-center gap-2 rounded-full border px-4 py-1 text-sm/6">
              <Mail className="size-4" />
              For technical questions, contact us by
              <a href="mailto:sailwebs@apecloud.com" className="underline">
                Email <span aria-hidden="true">&rarr;</span>
              </a>
            </div>
          </div>
          <div className="text-center">
            <h1 className="text-4xl font-semibold tracking-tight text-balance sm:text-5xl">
              Production-Ready RAG Platform with Graph, Vector & Full-Text
              Search
            </h1>
            <p className="text-muted-foreground mt-8 text-lg">
              ApeRAG is a production-ready RAG (Retrieval-Augmented Generation)
              platform that combines Graph RAG, vector search, and full-text
              search. Build sophisticated AI applications with hybrid retrieval,
              multimodal document processing, and enterprise-grade management
              features.
            </p>
            <div className="mt-10 flex items-center justify-center gap-x-6">
              <Button asChild>
                <Link href="/workspace/collections">Get started</Link>
              </Button>
              <Button asChild variant="outline">
                <Link href="/marketplace">
                  Marketplace <ArrowRight />
                </Link>
              </Button>
            </div>
          </div>
        </PageContent>
        <div
          aria-hidden="true"
          className="absolute inset-x-0 top-[calc(100%-30rem)] -z-10 transform-gpu overflow-hidden blur-3xl sm:top-[calc(100%-40rem)]"
        >
          <div
            style={{
              clipPath:
                'polygon(74.1% 44.1%, 100% 61.6%, 97.5% 26.9%, 85.5% 0.1%, 80.7% 2%, 72.5% 32.5%, 60.2% 62.4%, 52.4% 68.1%, 47.5% 58.3%, 45.2% 34.5%, 27.5% 76.7%, 0.1% 64.9%, 17.9% 100%, 27.6% 76.8%, 76.1% 97.7%, 74.1% 44.1%)',
            }}
            className="relative left-[calc(50%+3rem)] aspect-1155/678 w-144.5 -translate-x-1/2 bg-linear-to-tr from-[#ff80b5] to-[#9089fc] opacity-30 sm:left-[calc(50%+36rem)] sm:w-288.75"
          ></div>
        </div>

        <PageContent className="">
          <div className="mb-12 text-center text-5xl font-bold">
            Key Features
          </div>
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-2">
            {features.map((feature, index) => {
              return (
                <div
                  key={index}
                  className="relative overflow-hidden bg-gray-950/[4%] bg-[image:radial-gradient(var(--pattern-fg)_1px,_transparent_0)] bg-[size:10px_10px] bg-fixed [--pattern-fg:var(--color-gray-950)]/5 after:pointer-events-none after:absolute after:inset-0 after:rounded-lg after:inset-ring after:inset-ring-gray-950/5 dark:[--pattern-fg:var(--color-white)]/10 dark:after:inset-ring-white/10"
                >
                  <div className="p-8">
                    <h3 className="mb-4 text-2xl">{feature.title}</h3>
                    <div className="text-muted-foreground">
                      {feature.description}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </PageContent>
        <PageContent className="py-12"></PageContent>
      </PageContainer>
    </>
  );
}
