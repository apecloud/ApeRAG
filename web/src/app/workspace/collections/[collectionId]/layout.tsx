import { CollectionProvider } from '@/components/providers/collection-provider';
import { getServerApi } from '@/lib/api/server';
import { notFound } from 'next/navigation';

export default async function ChatLayout({
  params,
  children,
}: Readonly<{
  params: Promise<{ collectionId: string }>;
  children: React.ReactNode;
}>) {
  const { collectionId } = await params;
  const serverApi = await getServerApi();
  const [collectionres, shareRes] = await Promise.all([
    serverApi.defaultApi.collectionsCollectionIdGet({
      collectionId,
    }),
    serverApi.defaultApi.collectionsCollectionIdSharingGet({
      collectionId,
    }),
  ]);
  const collection = collectionres.data;

  if (!collection) {
    notFound();
  }

  return (
    <CollectionProvider collection={collection} share={shareRes.data}>
      {children}
    </CollectionProvider>
  );
}
