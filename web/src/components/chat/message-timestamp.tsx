import { ChatMessage } from '@/api';
import { cn } from '@/lib/utils';
import { useFormatter } from 'next-intl';

export const MessageTimestamp = ({
  parts,
  className,
}: React.ComponentProps<'div'> & { parts: ChatMessage[] }) => {
  const timestamp = parts.find((part) => part.timestamp)?.timestamp;
  const format = useFormatter();
  return (
    <div className={cn('text-muted-foreground text-xs', className)}>
      {timestamp && format.dateTime(new Date(timestamp * 1000), 'medium')}
    </div>
  );
};
