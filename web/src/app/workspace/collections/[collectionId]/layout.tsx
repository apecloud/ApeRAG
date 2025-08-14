import { CollectionViewStatusEnum } from '@/api';
import { FormatDate } from '@/components/format-date';
import { PageHeader } from '@/components/page-header';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { getServerApi } from '@/lib/api/server';
import _ from 'lodash';
import { Calendar, Trash } from 'lucide-react';

export default async function Layout({
  children,
  params,
}: Readonly<{
  children: React.ReactNode;
  params: Promise<{ collectionId: string }>;
}>) {
  const { collectionId } = await params;
  const serverApi = await getServerApi();

  const res = await serverApi.defaultApi.collectionsCollectionIdGet({
    collectionId,
  });
  const collection = res.data;
  const badgeColor: {
    [key in CollectionViewStatusEnum]: string;
  } = {
    ACTIVE: 'text-green-500',
    INACTIVE: 'text-red-500',
    DELETED: '',
  };

  return (
    <>
      <PageHeader
        title={collection.title}
        description={collection.description}
        breadcrumbs={[
          {
            title: 'Collections',
            href: '/workspace/collections',
          },
          {
            title: collection.title,
          },
        ]}
      >
        <div className="flex flex-row items-center gap-6">
          <div>
            {collection.created && (
              <div className="text-muted-foreground flex items-center gap-1 text-sm">
                <Calendar className="size-3" />
                <FormatDate datetime={new Date(collection.created)} />
              </div>
            )}
          </div>
          {collection.status && (
            <Badge
              variant="secondary"
              className={badgeColor[collection.status]}
            >
              {_.upperFirst(_.lowerCase(collection.status))}
            </Badge>
          )}
          <Button size="icon" variant="outline">
            <Trash className="text-red-500" />
          </Button>
        </div>
      </PageHeader>
      {children}
    </>
  );
}
