import { Metadata } from 'next';
import { getTranslations } from 'next-intl/server';

export async function generateMetadata(): Promise<Metadata> {
  const page_marketplace = await getTranslations('page_marketplace');
  return {
    title: page_marketplace('metadata.title'),
    description: page_marketplace('metadata.description'),
  };
}

export default async function MarketplaceLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return children;
}
