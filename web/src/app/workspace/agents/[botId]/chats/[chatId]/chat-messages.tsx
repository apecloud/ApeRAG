'use client';

import { ChatDetails, ChatMessage } from '@/api';
import { Markdown } from '@/components/markdown';
import { Card, CardContent } from '@/components/ui/card';
import { apiClient } from '@/lib/api/client';
import { Bot, UserRound } from 'lucide-react';
import { useParams } from 'next/navigation';
import { useCallback, useState } from 'react';

const UserMessage = ({ children }: { children: React.ReactNode }) => {
  return (
    <div className="ml-auto flex w-max max-w-[85%] flex-row gap-4">
      <div className="bg-primary text-primary-foreground rounded-lg px-3 py-2 text-sm">
        {children}
      </div>
      <div>
        <div className="bg-muted/85 text-muted-foreground flex size-11 flex-col justify-center rounded-full">
          <UserRound className="size-5 self-center" />
        </div>
      </div>
    </div>
  );
};

const AIMessage = ({ children }: { children: React.ReactNode }) => {
  return (
    <div className="flex w-max max-w-[85%] flex-row gap-4">
      <div>
        <div className="bg-primary text-primary-foreground/80 flex size-11 flex-col justify-center rounded-full">
          <Bot className="size-5 self-center" />
        </div>
      </div>
      <Card className="dark:border-none">
        <CardContent>{children}</CardContent>
      </Card>
    </div>
  );
};

export const ChatMessages = ({ chat }: { chat: ChatDetails }) => {
  const [messages, setMessages] = useState<Array<Array<ChatMessage>>>(
    chat.history || [],
  );
  const { botId, chatId } = useParams<{ botId: string; chatId: string }>();

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const loadChatDetail = useCallback(async () => {
    if (!botId || !chatId) return;
    const res = await apiClient.defaultApi.botsBotIdChatsChatIdGet({
      botId,
      chatId,
    });
    setMessages(res.data?.history || []);
  }, [botId, chatId]);

  if (messages.length === 0) {
    return <div>no messages found</div>;
  }

  return (
    <div className="flex flex-col gap-6">
      {messages?.map((parts, index) => {
        const isAI = parts.some((part) => part.role === 'ai');
        const content = (
          <Markdown>{parts.map((part) => part.data || '').join('')}</Markdown>
        );
        return isAI ? (
          <AIMessage key={index}>{content}</AIMessage>
        ) : (
          <UserMessage key={index}>{content}</UserMessage>
        );
      })}
    </div>
  );
};
