'use client';

import { ChatDetails, ChatMessage } from '@/api';
import { Markdown } from '@/components/markdown';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import { apiClient } from '@/lib/api/client';
import { Bot, ChevronRight, UserRound } from 'lucide-react';
import { useParams } from 'next/navigation';
import { useCallback, useEffect, useState } from 'react';

const UserMessage = ({ parts }: { parts: ChatMessage[] }) => {
  return (
    <div className="ml-auto flex w-max max-w-[85%] flex-row gap-4">
      <div className="bg-primary text-primary-foreground rounded-lg px-3 py-2 text-sm">
        {parts?.map((part) => part.data || '').join('')}
      </div>
      <div>
        <div className="bg-muted/85 text-muted-foreground flex size-11 flex-col justify-center rounded-full">
          <UserRound className="size-5 self-center" />
        </div>
      </div>
    </div>
  );
};

const AIMessage = ({ parts }: { parts: ChatMessage[] }) => {
  const parseToolCall = useCallback(
    (content: string): { title: string; body: string } => {
      const lines = content.split('\n');
      const firstLine = lines[0] || '';
      const titleMatch = firstLine.match(/^\*\*(.*?)\*\*$/);
      if (titleMatch) {
        const title = titleMatch[1].trim();
        const body = lines.slice(1).join('\n').trim();
        return { title, body };
      }
      return { title: 'Tool call', body: content };
    },
    [],
  );

  const getContent = useCallback(
    (part: ChatMessage) => {
      switch (part.type) {
        case 'thinking':

        case 'tool_call_result':
          const { title, body } = parseToolCall(part.data || '');
          return (
            <Collapsible className="group/collapsible my-2">
              <CollapsibleTrigger asChild>
                <Button
                  variant="secondary"
                  className="w-full cursor-pointer justify-start"
                >
                  <ChevronRight className="transition-transform duration-200 group-data-[state=open]/collapsible:rotate-90" />
                  <span className="block max-w-160 truncate">{title}</span>
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-2 rounded-md border p-4">
                <Markdown>{body}</Markdown>
              </CollapsibleContent>
            </Collapsible>
          );
        case 'message':
          return <Markdown>{part.data}</Markdown>;

        default:
          return part.data;
      }
    },
    [parseToolCall],
  );

  return (
    <div className="flex w-max max-w-[85%] flex-row gap-4">
      <div>
        <div className="bg-primary text-primary-foreground/80 flex size-11 flex-col justify-center rounded-full">
          <Bot className="size-5 self-center" />
        </div>
      </div>
      <Card className="dark:border-none">
        <CardContent>
          {parts.map((part, index) => (
            <div key={`${index}-${part.id}`}>{getContent(part)}</div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
};

export const ChatMessages = ({ chat }: { chat: ChatDetails }) => {
  const [messages, setMessages] = useState<Array<Array<ChatMessage>>>(
    chat.history || [],
  );
  const { botId, chatId } = useParams<{ botId: string; chatId: string }>();

  const loadChatDetail = useCallback(async () => {
    if (!botId || !chatId) return;
    const res = await apiClient.defaultApi.botsBotIdChatsChatIdGet({
      botId,
      chatId,
    });
    setMessages(res.data?.history || []);
  }, [botId, chatId]);

  useEffect(() => {
    loadChatDetail();
  }, [loadChatDetail]);

  if (messages.length === 0) {
    return <div>no messages found</div>;
  }

  return (
    <div className="flex flex-col gap-6">
      {messages?.map((parts, index) => {
        const isAI = parts.some((part) => part.role === 'ai');

        return isAI ? (
          <AIMessage key={index} parts={parts} />
        ) : (
          <UserMessage key={index} parts={parts} />
        );
      })}
    </div>
  );
};
