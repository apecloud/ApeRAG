import { Reference } from '@/api';
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
import { MessageCollapseContent } from './message-collapse-content';

export const MessageReference = ({
  references,
}: {
  references: Reference[];
}) => {
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
          <DrawerTitle className="font-bold">References</DrawerTitle>
        </DrawerHeader>
        <div className="overflow-auto px-4 pb-4 select-text">
          {references?.map((reference: Reference, index) => {
            return (
              <MessageCollapseContent
                defaultOpen={index <= 2}
                key={index}
                title={
                  <div className="flex flex-row justify-between">
                    <div>
                      {index + 1}.{' '}
                      {reference.metadata?.query ||
                        _.truncate(reference.text, { length: 30 })}
                    </div>
                    <div className="text-muted-foreground ml-auto flex flex-row items-center gap-2 text-xs">
                      <span>{_.startCase(reference.metadata?.type)}</span>
                      <span>{(reference.score || 0).toFixed(2)}</span>
                    </div>
                  </div>
                }
              >
                <Markdown>{reference.text}</Markdown>
              </MessageCollapseContent>
            );
          })}
        </div>
      </DrawerContent>
    </Drawer>
  );
};
