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
  let share;

  try {
    const [collectionRes, shareRes] = await Promise.all([
      serverApi.defaultApi.collectionsCollectionIdGet({
        collectionId,
      }),
      serverApi.defaultApi.collectionsCollectionIdSharingGet({
        collectionId,
      }),
    ]);
    collection = collectionRes.data;
    share = shareRes.data;
  } catch (err) {
    console.log(err);
  }

  if (!collection || !share) {
    notFound();
  }

  return (
    <CollectionProvider collection={collection} share={share}>
      {children}
    </CollectionProvider>
  );
}
