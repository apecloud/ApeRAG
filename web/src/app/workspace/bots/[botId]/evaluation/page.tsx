import { notFound } from 'next/navigation';

export default async function Page({
  params: _params,
}: {
  params: Promise<{ botId: string }>;
}) {
  notFound();
}
