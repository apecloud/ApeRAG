import { SearchResult } from '@/api';
import { Markdown } from '@/components/markdown';
import { Button } from '@/components/ui/button';
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
import { Slot } from '@radix-ui/react-slot';
import _ from 'lodash';
import { ChevronRight } from 'lucide-react';
import { useTranslations } from 'next-intl';
import { ReactNode, useState } from 'react';

export const SearchResultDrawer = ({
  children,
  result,
}: {
  children: ReactNode;
  result: SearchResult;
}) => {
  const [visible, setVisible] = useState<boolean>(false);
  const page_search = useTranslations('page_search');

  if (_.isEmpty(result.items)) {
    return children;
  }

  return (
    <Drawer
      direction="right"
      open={visible}
      onOpenChange={() => setVisible(false)}
      handleOnly={true}
    >
      <DrawerTrigger asChild>
        <Slot
          onClick={(e) => {
            setVisible(true);
            e.preventDefault();
          }}
        >
          {children}
        </Slot>
      </DrawerTrigger>
      <DrawerContent className="flex sm:min-w-xl md:min-w-2xl">
        <DrawerHeader>
          <DrawerTitle className="font-bold">
            {page_search('search_result')}
          </DrawerTitle>
        </DrawerHeader>
        <div className="overflow-auto px-4 pb-4 select-text">
          {result.items?.map((item, index) => {
            return (
              <Collapsible
                key={index}
                className="group/collapsible my-2"
                defaultOpen={index === 0}
              >
                <CollapsibleTrigger asChild>
                  <Button
                    variant="secondary"
                    className="w-full cursor-pointer justify-between"
                  >
                    <div className="flex flex-1 flex-row items-center gap-2">
                      <ChevronRight className="transition-transform duration-200 group-data-[state=open]/collapsible:rotate-90" />
                      <div className="block flex-1 text-left">
                        {item.rank}.{' '}
                        {item.source ||
                          _.truncate(item.content, { length: 30 })}
                      </div>
                    </div>
                    <div className="text-muted-foreground flex flex-row items-center gap-4 text-xs">
                      <span>{_.startCase(item.recall_type)}</span>
                      <span>{(item.score || 0).toFixed(2)}</span>
                    </div>
                  </Button>
                </CollapsibleTrigger>

                <CollapsibleContent className="mt-2 rounded-md border p-4">
                  <Markdown>{item.content}</Markdown>
                </CollapsibleContent>
              </Collapsible>
            );
          })}
        </div>
      </DrawerContent>
    </Drawer>
  );
};
