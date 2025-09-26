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

  let collection;

  try {
    const [collectionRes] = await Promise.all([
      serverApi.defaultApi.collectionsCollectionIdGet({
        collectionId,
      }),
    ]);
    collection = collectionRes.data;
  } catch (err) {
    console.log(err);
  }

  if (!collection) {
    notFound();
  }

  return (
    <CollectionProvider collection={collection}>{children}</CollectionProvider>
  );
}
