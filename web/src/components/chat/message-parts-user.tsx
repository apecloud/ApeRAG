import { ChatMessage } from '@/api';
import { Markdown } from '@/components/markdown';
import { useBotContext } from '@/components/providers/bot-provider';
import { UserRound } from 'lucide-react';
import { useMemo } from 'react';
import { MessageTimestamp } from './message-timestamp';

export const MessagePartsUser = ({ parts }: { parts: ChatMessage[] }) => {
  const { collections } = useBotContext();

  const message = useMemo(() => {
    const rawMessage = parts?.map((part) => part.data || '').join('') || '';
    if (!rawMessage || collections.length === 0) return rawMessage;

    const collectionNameById = new Map(
      collections
        .filter((collection) => collection.id)
        .map((collection) => [
          collection.id as string,
          collection.title || collection.id || '',
        ]),
    );

    const mentionPattern = /(^|\s)@([A-Za-z0-9]{24})(?=\s|$)/g;
    return rawMessage.replace(mentionPattern, (match, prefix, collectionId) => {
      const collectionName = collectionNameById.get(collectionId);
      if (!collectionName) return match;
      return `${prefix}@${collectionName}`;
    });
  }, [collections, parts]);

  return (
    <div className="ml-auto flex w-max flex-row gap-4">
      <div className="flex max-w-sm flex-col gap-2 sm:max-w-lg md:max-w-2xl lg:max-w-3xl xl:max-w-4xl">
        <div className="bg-primary text-primary-foreground rounded-lg p-4 text-sm">
          <Markdown>{message}</Markdown>
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
