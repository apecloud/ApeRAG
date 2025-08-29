import { Metadata } from 'next';
import { getTranslations } from 'next-intl/server';
import { SignUpForm } from './signup-form';

export async function generateMetadata(): Promise<Metadata> {
  const page_auth = await getTranslations('page_auth');
  return {
    title: page_auth('signup'),
  };
}

export default async function Page() {
  return <SignUpForm />;
}
