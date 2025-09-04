import { ChatMessage, Feedback } from '@/api';
import { CopyToClipboard } from '@/components/copy-to-clipboard';
import { Card } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { cn } from '@/lib/utils';
import _ from 'lodash';
import { Bot, LoaderCircle } from 'lucide-react';
import { useMemo } from 'react';
import { MessageFeedback } from './message-feedback';
import { MessagePartAi } from './message-part-ai';
import { MessageReference } from './message-reference';
import { MessageTimestamp } from './message-timestamp';

export const MessagePartsAi = ({
  pending,
  loading,
  parts,
  hanldeMessageFeedback,
}: {
  pending: boolean;
  loading: boolean;
  parts: ChatMessage[];
  hanldeMessageFeedback: (part: ChatMessage, feedback: Feedback) => void;
}) => {
  const references = useMemo(
    () => parts.findLast((part) => part.references)?.references || [],
    [parts],
  );

  return (
    <div className="flex w-max flex-row gap-4">
      <div>
        <div className="bg-muted text-muted-foreground relative flex size-12 flex-col justify-center rounded-full">
          {loading && (
            <LoaderCircle className="absolute -left-1 size-14 animate-spin opacity-20" />
          )}
          <Bot className={cn('size-6 self-center')} />
        </div>
      </div>
      <div className="flex max-w-sm flex-col gap-1 sm:max-w-lg md:max-w-2xl lg:max-w-3xl xl:max-w-4xl">
        <Card className="dark:border-card/0 block gap-0 px-4 py-4 text-sm">
          {pending ? (
            <div className="flex flex-row gap-2 py-2">
              <div className="bg-muted-foreground animate-caret-blink size-2 rounded-full delay-0"></div>
              <div className="bg-muted-foreground animate-caret-blink size-2 rounded-full delay-200"></div>
              <div className="bg-muted-foreground animate-caret-blink size-2 rounded-full delay-400"></div>
            </div>
          ) : (
            parts.map((part, index) => (
              <MessagePartAi
                key={`${index}-${part.id}`}
                part={part}
                loading={loading}
              />
            ))
          )}
        </Card>
        <div className="flex flex-row items-center gap-2">
          <MessageTimestamp parts={parts} className="mr-2" />
          <Separator
            orientation="vertical"
            className="data-[orientation=vertical]:h-4"
          />
          {!_.isEmpty(references) && (
            <>
              <MessageReference references={references} />
              <Separator
                orientation="vertical"
                className="data-[orientation=vertical]:h-4"
              />
            </>
          )}
          <MessageFeedback
            parts={parts}
            hanldeMessageFeedback={hanldeMessageFeedback}
          />
          <Separator
            orientation="vertical"
            className="data-[orientation=vertical]:h-4"
          />
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
