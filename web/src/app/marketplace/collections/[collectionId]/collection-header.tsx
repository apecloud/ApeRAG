'use client';

import { SharedCollection } from '@/api';
import { PageContent } from '@/components/page-container';
import { useAppContext } from '@/components/providers/app-provider';
import { Badge } from '@/components/ui/badge';
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
  CardAction,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { apiClient } from '@/lib/api/client';
import { cn } from '@/lib/utils';
import _ from 'lodash';
import { BookOpen, Files, Star, User, VectorSquare } from 'lucide-react';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { useCallback, useMemo } from 'react';
import { FaStar } from 'react-icons/fa6';

export const CollectionHeader = ({
  className,
  collection,
}: {
  className?: string;
  collection: SharedCollection;
}) => {
  const router = useRouter();
  const pathname = usePathname();

  const { user, signIn } = useAppContext();
  const page_collections = useTranslations('page_collections');
  const page_documents = useTranslations('page_documents');
  const page_marketplace = useTranslations('page_marketplace');
  const page_graph = useTranslations('page_graph');
  const sidebar_workspace = useTranslations('sidebar_workspace');

  const isOwner = useMemo(
    () => collection.owner_user_id === user?.id,
    [collection.owner_user_id, user?.id],
  );
  const isSubscriber = useMemo(
    () => collection.subscription_id !== null,
    [collection.subscription_id],
  );

  const handleSubscribe = useCallback(async () => {
    if (!user) {
      signIn();
      return;
    }

    if (isSubscriber) {
      await apiClient.defaultApi.marketplaceCollectionsCollectionIdSubscribeDelete(
        {
          collectionId: collection.id,
        },
      );
    } else {
      await apiClient.defaultApi.marketplaceCollectionsCollectionIdSubscribePost(
        {
          collectionId: collection.id,
        },
      );
    }
    router.refresh();
  }, [collection.id, isSubscriber, router, signIn, user]);

  return (
    <PageContent className={cn('flex flex-col gap-4 pb-0', className)}>
      <div className="flex items-center">
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink asChild>
                <Link
                  href="/marketplace"
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
          <CardAction className="text-muted-foreground flex flex-row items-center gap-4 text-xs">
            {isOwner ? (
              <Badge>{page_collections('mine')}</Badge>
            ) : (
              <div className="flex flex-row items-center gap-1">
                <User className="size-4" />
                <div className="max-w-60 truncate">
                  {collection.owner_username}
                </div>
              </div>
            )}
            <Button
              variant="outline"
              size="sm"
              hidden={isOwner}
              onClick={handleSubscribe}
              className="text-muted-foreground cursor-pointer text-xs"
            >
              {isSubscriber ? <FaStar className="text-orange-500" /> : <Star />}

              {isSubscriber
                ? page_collections('subscribed')
                : page_collections('subscribe')}
            </Button>
          </CardAction>
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
