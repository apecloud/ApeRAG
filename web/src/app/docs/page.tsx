import { getIndexPageUrl } from '@/lib/docs';
import { notFound, redirect } from 'next/navigation';

export default async function Page() {
  const indexPage = await getIndexPageUrl();
  if (indexPage) {
    redirect(indexPage);
  } else {
    notFound();
  }
}
