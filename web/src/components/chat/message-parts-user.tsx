import type { ChatMessage } from '@/features/bot/types';
import { Markdown } from '@/components/markdown';
import { useAppContext } from '@/components/providers/app-provider';
import { useBotContext } from '@/components/providers/bot-provider';
import { UserAvatar } from '@/components/user-avatar';
import { useMemo } from 'react';
import { MessageTimestamp } from './message-timestamp';

const isMentionBoundary = (value?: string) => {
  if (!value) return true;
  return !/[A-Za-z0-9_-]/.test(value);
};

const replaceCollectionMentions = (
  rawMessage: string,
  collectionNameById: Map<string, string>,
) => {
  const collectionIds = Array.from(collectionNameById.keys()).sort(
    (left, right) => right.length - left.length,
  );
  if (collectionIds.length === 0) return rawMessage;

  let result = '';
  let cursor = 0;

  while (cursor < rawMessage.length) {
    const mentionStart = rawMessage.indexOf('@', cursor);
    if (mentionStart === -1) {
      result += rawMessage.slice(cursor);
      break;
    }

    result += rawMessage.slice(cursor, mentionStart);

    if (!isMentionBoundary(rawMessage[mentionStart - 1])) {
      result += '@';
      cursor = mentionStart + 1;
      continue;
    }

    const matchedCollectionId = collectionIds.find((collectionId) => {
      const mentionEnd = mentionStart + 1 + collectionId.length;
      return (
        rawMessage.startsWith(collectionId, mentionStart + 1) &&
        isMentionBoundary(rawMessage[mentionEnd])
      );
    });

    if (!matchedCollectionId) {
      result += '@';
      cursor = mentionStart + 1;
      continue;
    }

    result += `@${collectionNameById.get(matchedCollectionId) || matchedCollectionId}`;
    cursor = mentionStart + 1 + matchedCollectionId.length;
  }

  return result;
};

export const MessagePartsUser = ({ parts }: { parts: ChatMessage[] }) => {
  const { collections } = useBotContext();
  const { user } = useAppContext();

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

    // Replace only exact @collection-id mentions that exist in the current bot context.
    return replaceCollectionMentions(rawMessage, collectionNameById);
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
        <UserAvatar user={user} className="size-12 text-base" />
      </div>
    </div>
  );
};
