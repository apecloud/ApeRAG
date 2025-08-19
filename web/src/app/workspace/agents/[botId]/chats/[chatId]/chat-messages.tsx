'use client';

import { ChatDetails, ChatMessage, Reference } from '@/api';
import { CopyToClipboard } from '@/components/copy-to-clipboard';
import { Markdown } from '@/components/markdown';
import { PageContent } from '@/components/page-container';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import {
  Drawer,
  DrawerContent,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
} from '@/components/ui/drawer';
import { useIsMobile } from '@/hooks/use-mobile';
import { apiClient } from '@/lib/api/client';
import { useWebSocket } from 'ahooks';
import { animateScroll as scroll } from 'react-scroll';

import { ReadyState } from 'ahooks/lib/useWebSocket';
import _ from 'lodash';
import {
  AlertCircleIcon,
  Bot,
  ChevronRight,
  Sparkles,
  UserRound,
} from 'lucide-react';
import { useFormatter } from 'next-intl';
import { useParams } from 'next/navigation';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { ChatInput, ChatInputSubmitParams } from './chat-input';

const CollapseContent = ({
  defaultOpen,
  title,
  children,
}: {
  defaultOpen?: boolean;
  title: React.ReactNode;
  children: React.ReactNode;
}) => {
  return (
    <Collapsible className="group/collapsible my-2" defaultOpen={defaultOpen}>
      <CollapsibleTrigger asChild>
        <Button variant="secondary" className="w-full cursor-pointer">
          <ChevronRight className="transition-transform duration-200 group-data-[state=open]/collapsible:rotate-90" />
          <div className="block flex-1 text-left">{title}</div>
        </Button>
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-2 rounded-md border p-4">
        {children}
      </CollapsibleContent>
    </Collapsible>
  );
};

const ReferenceContent = ({ parts }: { parts: ChatMessage[] }) => {
  const references = parts.findLast((part) => part.references)?.references;
  if (_.isEmpty(references)) {
    return;
  }
  return (
    <Drawer direction="right" handleOnly={true}>
      <DrawerTrigger asChild>
        <Button variant="ghost" size="icon">
          <Badge
            className="h-5 min-w-5 rounded-full px-1 font-mono tabular-nums"
            variant="destructive"
          >
            {references?.length}
          </Badge>
        </Button>
      </DrawerTrigger>
      <DrawerContent className="flex min-w-2xl">
        <DrawerHeader>
          <DrawerTitle className="font-bold">References</DrawerTitle>
        </DrawerHeader>
        <div className="overflow-auto px-4 pb-4 select-text">
          {references?.map((reference: Reference, index) => {
            return (
              <CollapseContent
                defaultOpen={index <= 2}
                key={index}
                title={
                  <div className="flex flex-row justify-between">
                    <div>
                      {reference.metadata?.type ||
                        reference.metadata?.query ||
                        _.truncate(reference.text, { length: 30 })}
                    </div>
                    <div className="ml-auto flex flex-row items-center gap-2">
                      <Sparkles className="text-muted-foreground size-4" />
                      <span>{(reference.score || 0).toFixed(2)}</span>
                    </div>
                  </div>
                }
              >
                <Markdown>{reference.text}</Markdown>
              </CollapseContent>
            );
          })}
        </div>
      </DrawerContent>
    </Drawer>
  );
};

const MessageTimestamp = ({ parts }: { parts: ChatMessage[] }) => {
  const timestamp = parts.find((part) => part.timestamp)?.timestamp;
  const format = useFormatter();
  return (
    <div className="text-muted-foreground text-xs">
      {timestamp && format.dateTime(new Date(timestamp), 'medium')}
    </div>
  );
};

const UserMessage = ({ parts }: { parts: ChatMessage[] }) => {
  return (
    <div className="ml-auto flex w-max max-w-[85%] flex-row gap-4">
      <div className="flex flex-col gap-2">
        <div className="bg-primary text-primary-foreground rounded-lg px-4 py-3 text-sm">
          {parts?.map((part) => part.data || '').join('')}
        </div>
        <MessageTimestamp parts={parts} />
      </div>
      <div>
        <div className="bg-muted text-muted-foreground flex size-10 flex-col justify-center rounded-full">
          <UserRound className="size-5 self-center" />
        </div>
      </div>
    </div>
  );
};

const AIMessagePart = ({ part }: { part: ChatMessage }) => {
  const parseToolCall = useCallback(
    (content: string): { title: string; body: string } => {
      const lines = content.split('\n');
      const firstLine = lines[0] || '';
      const titleMatch = firstLine.match(/^\*\*(.*?)\*\*$/);
      if (titleMatch) {
        const title = _.truncate(titleMatch[1].trim(), { length: 100 });
        const body = lines.slice(1).join('\n').trim();
        return { title, body };
      }
      return { title: 'Tool call', body: content };
    },
    [],
  );
  switch (part.type) {
    case 'error':
      return (
        <Alert variant="destructive">
          <AlertCircleIcon />
          <AlertDescription>{part.data}</AlertDescription>
        </Alert>
      );
    case 'thinking':
      return (
        <CollapseContent title="Thinging">
          <Markdown>{part.data}</Markdown>
        </CollapseContent>
      );
    case 'tool_call_result':
      const { title, body } = parseToolCall(part.data || '');
      return (
        <CollapseContent title={title}>
          <Markdown>{body}</Markdown>
        </CollapseContent>
      );
    case 'message':
      return <Markdown>{part.data}</Markdown>;
    default:
      return part.data;
  }
};

const AIMessage = ({ parts }: { parts: ChatMessage[] }) => {
  return (
    <div className="flex w-max max-w-[85%] flex-row gap-4">
      <div>
        <div className="bg-muted text-muted-foreground flex size-10 flex-col justify-center rounded-full">
          <Bot className="size-5 self-center" />
        </div>
      </div>
      <div className="flex flex-col gap-1">
        <Card className="py-4 dark:border-none">
          <CardContent className="px-4">
            {parts.map((part, index) => (
              <AIMessagePart key={`${index}-${part.id}`} part={part} />
            ))}
          </CardContent>
        </Card>
        <div className="flex flex-row items-center gap-2">
          <MessageTimestamp parts={parts} />
          <ReferenceContent parts={parts} />
          <CopyToClipboard
            variant="ghost"
            className="text-muted-foreground"
            text={parts.map((part) => part.data).join('')}
          />
        </div>
      </div>
    </div>
  );
};

export const ChatMessages = ({ chat }: { chat: ChatDetails }) => {
  const isMobile = useIsMobile();
  const [messages, setMessages] = useState<Array<Array<ChatMessage>>>(
    chat.history || [],
  );
  const [loading, setLoading] = useState<boolean>(false);
  const { botId, chatId } = useParams<{ botId: string; chatId: string }>();
  const { protocol, host } = useMemo(() => {
    if (typeof window !== 'undefined') {
      return {
        protocol: window.location.protocol === 'http:' ? 'ws://' : 'wss://',
        host: window.location.host,
      };
    } else {
      return {
        protocol: 'ws://',
        host: 'localhost:8000',
      };
    }
  }, []);

  const { sendMessage, readyState, disconnect, connect } = useWebSocket(
    `${protocol}${host}/api/v1/bots/${botId}/chats/${chatId}/connect`,
    {
      onMessage: (message) => {
        const fragment = JSON.parse(message.data) as ChatMessage;
        if (fragment.type === 'start') {
          setLoading(true);
        }
        if (fragment.type === 'stop') {
          setLoading(false);
        }

        setMessages((msgs) => {
          const parts = msgs.findLast((parts) => {
            return Boolean(
              parts.find(
                (part) => part.id === fragment.id && part.role === 'ai',
              ),
            );
          });

          if (parts) {
            if (fragment.type === 'stop' && Array.isArray(fragment.data)) {
              parts.push({
                type: 'references',
                references: fragment.data as Reference[],
                data: '',
                role: 'ai',
              });
            } else {
              const part = parts.find((p) => {
                if (fragment.type === 'message') {
                  return p.type === 'message';
                } else {
                  return fragment.part_id && fragment.part_id === p.part_id;
                }
              });
              if (part) {
                part.data = (part.data || '') + fragment.data;
              } else {
                parts.push(fragment);
              }
            }
          } else {
            msgs.push([
              {
                ...fragment,
                role: 'ai',
              },
            ]);
          }
          return [...msgs];
        });
      },
    },
  );

  const loadChatDetail = useCallback(async () => {
    if (!botId || !chatId) return;
    const res = await apiClient.defaultApi.botsBotIdChatsChatIdGet({
      botId,
      chatId,
    });
    setMessages(res.data?.history || []);
  }, [botId, chatId]);

  const handleSendMessage = useCallback(
    (params: ChatInputSubmitParams) => {
      const timestamp = Math.floor(new Date().getTime() / 1000);
      const part: ChatMessage = {
        type: 'message',
        role: 'human',
        data: params.query,
        timestamp,
      };
      setMessages((msgs) => {
        msgs?.push([part]);
        return [...msgs];
      });

      sendMessage(JSON.stringify(params));
    },
    [sendMessage],
  );

  const handleCancel = useCallback(() => {
    disconnect();
    connect();
    setLoading(false);
  }, [connect, disconnect]);

  useEffect(() => {
    loadChatDetail();
  }, [loadChatDetail]);

  useEffect(() => {
    scroll.scrollToBottom({ duration: 0 });
  }, [messages, chat]);

  return (
    <>
      <div className="flex flex-col gap-6 pb-80">
        {messages?.map((parts, index) => {
          const isAI = parts.some((part) => part.role === 'ai');

          return isAI ? (
            <AIMessage key={index} parts={parts} />
          ) : (
            <UserMessage key={index} parts={parts} />
          );
        })}
      </div>
      <div
        className={`fixed ${isMobile ? 'left-0' : 'left-[var(--sidebar-width)]'} bg-background right-0 bottom-0 z-10`}
      >
        <PageContent className="max-w-5xl pb-12">
          <ChatInput
            onSubmit={handleSendMessage}
            disabled={readyState !== ReadyState.Open}
            loading={loading}
            onCancel={handleCancel}
          />
        </PageContent>
      </div>
    </>
  );
};
