import type { Reference } from '@/features/bot/types';
import { Markdown } from '@/components/markdown';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Drawer,
  DrawerContent,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
} from '@/components/ui/drawer';
import _ from 'lodash';
import { useTranslations } from 'next-intl';
import { MessageCollapseContent } from './message-collapse-content';

export const MessageReference = ({
  references,
}: {
  references: Reference[];
}) => {
  const page_chat = useTranslations('page_chat');
  return (
    <Drawer direction="right" handleOnly={true}>
      <DrawerTrigger asChild>
        <Button variant="ghost" size="icon" className="cursor-pointer">
          <Badge
            className="h-5 min-w-5 rounded-full px-1 font-mono tabular-nums"
            variant="destructive"
          >
            {references?.length}
          </Badge>
        </Button>
      </DrawerTrigger>
      <DrawerContent className="flex sm:min-w-xl md:min-w-2xl">
        <DrawerHeader>
          <DrawerTitle className="font-bold">
            {page_chat('references')}
          </DrawerTitle>
        </DrawerHeader>
        <div className="overflow-auto px-4 pb-4 select-text">
          {references?.map((reference: Reference, index) => {
            const query =
              typeof reference.metadata?.query === 'string'
                ? reference.metadata.query
                : undefined;
            const type =
              typeof reference.metadata?.type === 'string'
                ? reference.metadata.type
                : undefined;
            const text = reference.text ?? '';
            return (
              <MessageCollapseContent
                defaultOpen={index <= 2}
                key={index}
                title={
                  <div className="flex flex-row justify-between">
                    <div>
                      {index + 1}.{' '}
                      {query || _.truncate(text, { length: 30 })}
                    </div>
                    <div className="text-muted-foreground ml-auto flex flex-row items-center gap-2 text-xs">
                      <span>{_.startCase(type)}</span>
                      <span>{(reference.score || 0).toFixed(2)}</span>
                    </div>
                  </div>
                }
              >
                <Markdown>{text}</Markdown>
              </MessageCollapseContent>
            );
          })}
        </div>
      </DrawerContent>
    </Drawer>
  );
};
