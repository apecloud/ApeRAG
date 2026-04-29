import { notFound } from 'next/navigation';

export default async function Page({
  params: _params,
}: Readonly<{
  params: Promise<{ collectionId: string; datasetId: string }>;
}>) {
  notFound();
}
