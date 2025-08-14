import { AppTopbar } from '@/components/app-topbar';
import Link from 'next/link';

export default async function Home() {
  return (
    <>
      <AppTopbar className="bg-background/60" />
      <div className="relative px-6 pt-8">
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
        <div className="mx-auto max-w-300 py-48">
          <div className="hidden sm:mb-8 sm:flex sm:justify-center">
            <div className="relative rounded-full border px-4 py-1 text-sm/6">
              Announcing our next round of funding.
              <a href="#" className="text-primary font-semibold">
                <span aria-hidden="true" className="absolute inset-0"></span>
                Read more <span aria-hidden="true">&rarr;</span>
              </a>
            </div>
          </div>
          <div className="text-center">
            <h1 className="text-5xl font-semibold tracking-tight text-balance sm:text-7xl">
              Best choice for building your own Knowledge Graph and for Context
              Engineering
            </h1>
            <p className="text-muted-foreground mt-8 sm:text-xl/8">
              ApeRAG is a production-ready RAG (Retrieval-Augmented Generation)
              platform that combines Graph RAG, vector search, and full-text
              search. Build sophisticated AI applications with hybrid retrieval,
              multimodal document processing, and enterprise-grade management
              features.
            </p>
            <div className="mt-10 flex items-center justify-center gap-x-6">
              <Link
                href="/workspace/collections"
                className="bg-primary text-accent hover:bg-primary/90 focus-visible:bg-primary/90 rounded-md px-3.5 py-2.5 text-sm font-semibold shadow-xs focus-visible:outline-2 focus-visible:outline-offset-2"
              >
                Get started
              </Link>
              <Link href="/" className="text-sm/6 font-semibold">
                Learn more <span aria-hidden="true">→</span>
              </Link>
            </div>
          </div>
        </div>
        <div
          aria-hidden="true"
          className="absolute inset-x-0 top-[calc(100%-13rem)] -z-10 transform-gpu overflow-hidden blur-3xl sm:top-[calc(100%-30rem)]"
        >
          <div
            style={{
              clipPath:
                'polygon(74.1% 44.1%, 100% 61.6%, 97.5% 26.9%, 85.5% 0.1%, 80.7% 2%, 72.5% 32.5%, 60.2% 62.4%, 52.4% 68.1%, 47.5% 58.3%, 45.2% 34.5%, 27.5% 76.7%, 0.1% 64.9%, 17.9% 100%, 27.6% 76.8%, 76.1% 97.7%, 74.1% 44.1%)',
            }}
            className="relative left-[calc(50%+3rem)] aspect-1155/678 w-144.5 -translate-x-1/2 bg-linear-to-tr from-[#ff80b5] to-[#9089fc] opacity-30 sm:left-[calc(50%+36rem)] sm:w-288.75"
          ></div>
        </div>
      </div>
    </>
  );
}
