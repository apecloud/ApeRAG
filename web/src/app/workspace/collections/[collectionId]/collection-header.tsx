import { Collection, CollectionViewStatusEnum } from '@/api';
import { FormatDate } from '@/components/format-date';
import { PageContent, PageTitle } from '@/components/page-container';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import _ from 'lodash';

import { Calendar, Trash } from 'lucide-react';

export const CollectionHeader = ({
  collection,
}: {
  collection: Collection;
}) => {
  const badgeColor: {
    [key in CollectionViewStatusEnum]: string;
  } = {
    ACTIVE: 'bg-green-500',
    INACTIVE: 'bg-red-500',
    DELETED: 'bg-gray-500',
  };

  return (
    <PageContent className="mb-4">
      <PageTitle>{collection.title}</PageTitle>
      <div className="flex flex-row items-center justify-between">
        <div className="flex flex-row items-center gap-6">
          <div>
            {collection.created && (
              <div className="text-muted-foreground flex items-center gap-1 text-sm">
                <Calendar className="size-3" />
                <FormatDate datetime={new Date(collection.created)} />
              </div>
            )}
          </div>
          <div className="flex items-center gap-2">
            <div
              className={cn(
                'size-2 rounded-2xl',
                collection.status
                  ? badgeColor[collection.status]
                  : 'bg-gray-500',
              )}
            />
            <div className="text-muted-foreground text-sm">
              {_.upperFirst(_.lowerCase(collection.status))}
            </div>
          </div>
        </div>
        <Button size="icon" variant="outline">
          <Trash className="text-red-500" />
        </Button>
      </div>
    </PageContent>
  );
};
