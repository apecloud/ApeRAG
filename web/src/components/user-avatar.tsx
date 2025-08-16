import { User } from '@/api';
// import { Avatar, AvatarFallback, AvatarImage } from '@radix-ui/react-avatar';

import { FaCircleUser } from 'react-icons/fa6';

export const UserAvatar = ({ }: { user?: User }) => {
  const UserIcon = () => (
    <FaCircleUser className="text-muted-foreground/50 size-8" />
  );

  return <UserIcon />

  // return user?.image ? (
  //   <Avatar className="h-8 w-8 overflow-hidden rounded-4xl">
  //     <AvatarImage src={user.image} />
  //     <AvatarFallback>
  //       <UserIcon />
  //     </AvatarFallback>
  //   </Avatar>
  // ) : (
  //   <UserIcon />
  // );
};

export const UserAvatarProfile = ({ user }: { user?: User }) => {
  const username = user?.username || user?.email?.split('@')[0];
  return (
    <div className="flex items-center gap-2 text-left text-sm">
      <UserAvatar user={user} />
      <div className="grid flex-1 text-left text-sm leading-tight">
        <span className="truncate font-medium">{username}</span>
        <span className="text-muted-foreground truncate text-xs">
          {user?.email}
        </span>
      </div>
    </div>
  );
};
