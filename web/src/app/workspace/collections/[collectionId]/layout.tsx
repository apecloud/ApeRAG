import { CollectionProvider } from '@/components/providers/collection-provider';
import {
  getCollection,
  getCollectionSharingStatus,
} from '@/features/collection/server-api';
import { notFound } from 'next/navigation';

export default async function ChatLayout({
  params,
  children,
}: Readonly<{
  params: Promise<{ collectionId: string }>;
  children: React.ReactNode;
}>) {
  const { collectionId } = await params;

  let collection;
  let share;

  try {
    const [collectionRes, shareRes] = await Promise.all([
      getCollection(collectionId),
      getCollectionSharingStatus(collectionId),
    ]);
    collection = collectionRes ?? undefined;
    share = shareRes ?? undefined;
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
