'use client';

import { SharedCollection } from '@/api';
import { PageContent } from '@/components/page-container';
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { cn } from '@/lib/utils';
import _ from 'lodash';
import { BookOpen, Files, VectorSquare } from 'lucide-react';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

export const CollectionHeader = ({
  className,
  collection,
}: {
  className?: string;
  collection: SharedCollection;
}) => {
  const pathname = usePathname();

  const page_documents = useTranslations('page_documents');
  const page_marketplace = useTranslations('page_marketplace');
  const page_graph = useTranslations('page_graph');
  const sidebar_workspace = useTranslations('sidebar_workspace');

  return (
    <PageContent className={cn('flex flex-col gap-4 pb-0', className)}>
      <div className="flex items-center">
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink asChild>
                <Link
                  href="/marketplace/collections"
                  className="text-foreground flex flex-row items-center gap-1"
                >
                  {page_marketplace('metadata.title')}
                </Link>
              </BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>{collection.title}</BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>

        <Button className="ml-auto" asChild>
          <Link href="/workspace/collections">
            <BookOpen />

            {sidebar_workspace('collections')}
          </Link>
        </Button>
      </div>

      <Card className="gap-0 p-0">
        <CardHeader className="p-4 pb-0">
          <CardTitle className="mb-0 text-2xl">{collection.title}</CardTitle>
        </CardHeader>
        <CardDescription className="mb-4 px-4">
          {_.truncate(collection.description || 'No description available', {
            length: 180,
          })}
        </CardDescription>
        <Separator />
        <div className="bg-accent/50 flex flex-row gap-2 rounded-b-xl px-4">
          <Button
            asChild
            data-active={Boolean(
              pathname.match(
                `/marketplace/collections/${collection.id}/documents`,
              ),
            )}
            className="hover:border-b-primary data-[active=true]:border-b-primary h-10 rounded-none border-y-2 border-y-transparent px-1 has-[>svg]:px-2"
            variant="ghost"
          >
            <Link href={`/marketplace/collections/${collection.id}/documents`}>
              <Files />
              <span className="hidden sm:inline">
                {page_documents('metadata.title')}
              </span>
            </Link>
          </Button>

          {collection.config?.enable_knowledge_graph && (
            <Button
              asChild
              data-active={Boolean(
                pathname.match(
                  `/marketplace/collections/${collection.id}/graph`,
                ),
              )}
              className="hover:border-b-primary data-[active=true]:border-b-primary h-10 rounded-none border-y-2 border-y-transparent px-1 has-[>svg]:px-2"
              variant="ghost"
            >
              <Link href={`/marketplace/collections/${collection.id}/graph`}>
                <VectorSquare />
                <span className="hidden sm:inline">
                  {page_graph('metadata.title')}
                </span>
              </Link>
            </Button>
          )}
        </div>
      </Card>
    </PageContent>
  );
};
