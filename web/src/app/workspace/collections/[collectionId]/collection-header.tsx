'use client';
import { Collection, CollectionViewStatusEnum } from '@/api';
import { FormatDate } from '@/components/format-date';
import { PageContent } from '@/components/page-container';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardAction,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
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
  Settings,
  Trash,
  VectorSquare,
} from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useMemo } from 'react';

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

  const pathname = usePathname();

  const urls = useMemo(() => {
    return {
      documents: `/workspace/collections/${collection.id}/documents`,
      search: `/workspace/collections/${collection.id}/search`,
      graph: `/workspace/collections/${collection.id}/graph`,
    };
  }, [collection.id]);

  return (
    <PageContent className="pb-0">
      <Card className="bg-none shadow-none">
        <CardHeader>
          <CardTitle className="text-2lg">{collection.title}</CardTitle>
          <CardDescription className="flex flex-row items-center gap-6">
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
          </CardDescription>
          <CardAction className="flex flex-row gap-1">
            <Button
              asChild
              size="sm"
              variant={urls.documents.match(pathname) ? 'secondary' : 'ghost'}
            >
              <Link href={urls.documents}>
                <Files />
                <span className="hidden lg:inline">Documents Explorer</span>
              </Link>
            </Button>

            <Button
              asChild
              size="sm"
              variant={urls.graph.match(pathname) ? 'secondary' : 'ghost'}
            >
              <Link href={`/workspace/collections/${collection.id}/graph`}>
                <VectorSquare />
                <span className="hidden lg:inline">Knowledge Graph</span>
              </Link>
            </Button>

            <Button
              asChild
              size="sm"
              variant={urls.search.match(pathname) ? 'secondary' : 'ghost'}
            >
              <Link href={`/workspace/collections/${collection.id}/search`}>
                <FolderSearch />
                <span className="hidden lg:inline">Experience Search</span>
              </Link>
            </Button>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button size="icon" variant="ghost">
                  <EllipsisVertical />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-46">
                <DropdownMenuItem asChild>
                  <Link
                    href={`/workspace/collections/${collection.id}/settings`}
                  >
                    <Settings /> Settings
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem variant="destructive">
                  <Trash /> Delete Collection
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </CardAction>
        </CardHeader>
      </Card>
    </PageContent>
  );
};
