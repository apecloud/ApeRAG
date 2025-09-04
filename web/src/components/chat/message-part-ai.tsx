import { ChatMessage } from '@/api';
import { Markdown } from '@/components/markdown';
import { Alert, AlertDescription } from '@/components/ui/alert';
import _ from 'lodash';
import { AlertCircleIcon } from 'lucide-react';
import { useCallback } from 'react';
import { MessageCollapseContent } from './message-collapse-content';

export const MessagePartAi = ({
  part,
  loading,
}: {
  part: ChatMessage;
  loading: boolean;
}) => {
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
        <MessageCollapseContent title="Thinging" animate={loading}>
          <Markdown>{part.data}</Markdown>
        </MessageCollapseContent>
      );
    case 'tool_call_result':
      const { title, body } = parseToolCall(part.data || '');
      return (
        <MessageCollapseContent title={title} animate={loading}>
          <Markdown>{body}</Markdown>
        </MessageCollapseContent>
      );
    case 'message':
      return <Markdown>{part.data}</Markdown>;
    case 'stop':
      return '';
    default:
      return part.data;
  }
};
