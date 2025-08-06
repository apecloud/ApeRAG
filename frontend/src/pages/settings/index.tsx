import { Navigate, useModel } from 'umi';

export default () => {
  const { user } = useModel('user');
  
  // Admin users go to user management, others go to model configuration
  const defaultPath = user?.role === 'admin' ? '/settings/users' : '/settings/modelConfiguration';
  
  return <Navigate to={defaultPath} replace />;
};
