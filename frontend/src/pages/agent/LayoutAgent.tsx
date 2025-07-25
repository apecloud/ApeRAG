import { Outlet, useModel } from 'umi';

export const LayoutAgent = () => {
  const { bot } = useModel('bot');

  if (!bot) return;
  return <Outlet />;
};
