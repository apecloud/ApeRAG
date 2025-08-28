import { ChatMessage } from '@/api';
import { Markdown } from '@/components/markdown';
import { UserRound } from 'lucide-react';
import { MessageTimestamp } from './message-timestamp';

export const MessagePartsUser = ({ parts }: { parts: ChatMessage[] }) => {
  return (
    <div className="ml-auto flex w-max flex-row gap-4">
      <div className="flex max-w-sm flex-col gap-2 sm:max-w-lg md:max-w-2xl lg:max-w-3xl xl:max-w-4xl">
        <div className="bg-primary text-primary-foreground rounded-lg p-4 text-sm">
          <Markdown>{parts?.map((part) => part.data || '').join('')}</Markdown>
        </div>
        <MessageTimestamp parts={parts} />
      </div>
      <div>
        <div className="bg-muted text-muted-foreground flex size-12 flex-col justify-center rounded-full">
          <UserRound className="size-5 self-center" />
        </div>
      </div>
    </div>
  );
};
