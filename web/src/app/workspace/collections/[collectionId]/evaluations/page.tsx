import { redirect } from 'next/navigation';

export default async function Page({
  params,
}: {
  params: Promise<{ collectionId: string }>;
}) {
  const { collectionId } = await params;
  redirect(`/workspace/collections/${collectionId}/benchmarks`);
}
