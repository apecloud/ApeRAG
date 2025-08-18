import { Collection, CollectionViewStatusEnum } from '@/api';
import { FormatDate } from '@/components/format-date';
import { PageContent, PageTitle } from '@/components/page-container';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { cn } from '@/lib/utils';
import _ from 'lodash';

import {
  Calendar,
  EllipsisVertical,
  Files,
  FolderSearch,
  MonitorUp,
  Settings,
  TextSearch,
  Trash,
  VectorSquare,
} from 'lucide-react';
import Link from 'next/link';

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
        <div className="flex flex-row items-center gap-2">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline">
                <EllipsisVertical />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-46">
              <DropdownMenuItem asChild>
                <Link href={`/workspace/collections/${collection.id}/files`}>
                  <Files /> File Explorer
                </Link>
              </DropdownMenuItem>
              <DropdownMenuItem asChild>
                <Link href={`/workspace/collections/${collection.id}/search`}>
                  <FolderSearch /> Experience Search
                </Link>
              </DropdownMenuItem>
              <DropdownMenuItem asChild>
                <Link href={`/workspace/collections/${collection.id}/search`}>
                  <VectorSquare /> Knowledge Graph
                </Link> 
              </DropdownMenuItem>
              <DropdownMenuItem asChild>
                <Link href={`/workspace/collections/${collection.id}/settings`}>
                  <Settings /> Settings
                </Link>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem variant="destructive">
                <Trash /> Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </PageContent>
  );
};
