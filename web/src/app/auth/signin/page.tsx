import { getAuthConfig } from '@/features/auth/server-api';
import { Metadata } from 'next';
import { getTranslations } from 'next-intl/server';
import { SignInForm } from './signin-form';

export async function generateMetadata(): Promise<Metadata> {
  const page_auth = await getTranslations('page_auth');
  return {
    title: page_auth('signin'),
  };
}

export default async function Page() {
  const config = await getAuthConfig();
  const methods = config.login_methods || ['local'];

  return <SignInForm methods={methods} />;
}
