import type { User } from '@/features/identity/types';
import { cn } from '@/lib/utils';

const avatarPalette = [
  { background: '#E8EEF8', foreground: '#28466F' },
  { background: '#E7F0EA', foreground: '#2F5C46' },
  { background: '#F2EBDD', foreground: '#6A4B24' },
  { background: '#F3E7E3', foreground: '#7A3F31' },
  { background: '#EAE7F2', foreground: '#4E4275' },
  { background: '#E5EEF0', foreground: '#315A61' },
] as const;

const getAvatarSeed = (user?: User) =>
  user?.username || user?.email?.split('@')[0] || user?.email || '';

const getAvatarText = (seed: string) => {
  const normalized = seed.trim();
  if (!normalized) return '?';

  const [first = '', second = ''] = normalized
    .split(/[\s._-]+/)
    .filter(Boolean);

  if (second) return `${first[0] ?? ''}${second[0] ?? ''}`.toUpperCase();
  return Array.from(first)[0]?.toUpperCase() ?? '?';
};

const getPaletteIndex = (seed: string) => {
  let hash = 0;
  for (const char of seed) {
    hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  }
  return hash % avatarPalette.length;
};

export const UserAvatar = ({
  user,
  className,
}: {
  user?: User;
  className?: string;
}) => {
  const seed = getAvatarSeed(user);
  const palette = avatarPalette[getPaletteIndex(seed)];

  return (
    <div
      className={cn(
        'flex size-8 shrink-0 items-center justify-center rounded-full text-sm font-semibold',
        className,
      )}
      style={{
        backgroundColor: palette.background,
        color: palette.foreground,
      }}
    >
      {getAvatarText(seed)}
    </div>
  );
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
