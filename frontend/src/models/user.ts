import { User } from '@/api';
import { api } from '@/services';
import { User as OidcUser } from 'oidc-react';
import { useEffect, useState } from 'react';
import { useModel } from 'umi';

/**
 * current login user
 */

let id_token: string | undefined;

export const getAuthorizationHeader = () => {
  if (id_token)
    return {
      Authorization: `Bearer ${id_token}`,
    };
};

export default () => {
  const [user, setUser] = useState<User>();
  const [userLoading, setUserLoading] = useState<boolean>(false);
  const [oidcUser, setOidcUser] = useState<OidcUser>();
  const { setLoading } = useModel('global');

  const getUser = async () => {
    setUserLoading(true);
    try {
      const res = await api.userGet();
      setUser({
        username: oidcUser?.profile.nickname || oidcUser?.profile.name || res.data.username || 'Guest',
        email: oidcUser?.profile.email || res.data.email,
        ...res.data,
      });
    } catch (error) {
      console.error('Failed to get user:', error);
    }
    setUserLoading(false);
  };

  useEffect(() => {
    // Try to get user info regardless of OIDC status
    // This handles cookie-based authentication
    getUser();
    
    if (oidcUser) {
      id_token = oidcUser?.id_token;
    }
  }, [oidcUser]);

  // Also try to get user on component mount
  useEffect(() => {
    getUser();
  }, []);

  useEffect(() => setLoading(userLoading), [userLoading]);

  return {
    oidcUser,
    setOidcUser,

    getUser,
    user,
    userLoading,
  };
};
