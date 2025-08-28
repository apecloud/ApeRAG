import { getServerApi } from '@/lib/api/server';
import { SignInForm } from './signin-form';

export default async function Page() {
  const apiServer = await getServerApi();
  let methods;
  try {
    const res = await apiServer.defaultApi.configGet();
    methods = res.data.login_methods || [];
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
  } catch (err) {
    methods = ['local'];
  }

  return <SignInForm methods={methods} />;
}
