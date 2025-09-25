'use client';

import { Bot, SharedBot } from '@/api';
import { FormatDate } from '@/components/format-date';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { apiClient } from '@/lib/api/client';
import { useInterval } from 'ahooks';
import { Calendar, Plus } from 'lucide-react';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { useCallback, useState } from 'react';

const BotCard = ({ bot }: { bot: Bot }) => {
  const page_bot = useTranslations('page_bot');
  return (
    <Link key={bot.id} href={`/workspace/bots/${bot.id}/chats`}>
      <Card className="hover:bg-accent/30 cursor-pointer gap-2 rounded-md">
        <CardHeader className="px-4">
          <CardTitle className="h-5 truncate">{bot.title}</CardTitle>
        </CardHeader>
        <CardDescription className="mb-4 truncate px-4">
          {bot.description || page_bot('no_description_available')}
        </CardDescription>
        <CardFooter className="justify-between px-4 text-xs">
          <div className="text-muted-foreground">
            {bot.created && (
              <div className="flex items-center gap-2">
                <Calendar className="size-3" />
                <FormatDate datetime={new Date(bot.created)} />
              </div>
            )}
          </div>
        </CardFooter>
      </Card>
    </Link>
  );
};

const BotsCardList = ({
  bots,
  searchValue,
}: {
  bots: Bot[];
  searchValue: string;
}) => {
  const page_bot = useTranslations('page_bot');
  return bots.length === 0 ? (
    <div className="bg-accent/50 text-muted-foreground rounded-lg py-40 text-center">
      {page_bot('no_bots_found')}
    </div>
  ) : (
    <div className="sm:grid-col-1 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {bots
        .filter((bot) => {
          if (searchValue === '') return true;
          return (
            bot.title?.match(new RegExp(searchValue)) ||
            bot.description?.match(new RegExp(searchValue))
          );
        })
        .map((bot) => {
          return <BotCard key={bot.id} bot={bot} />;
        })}
    </div>
  );
};

export const BotList = ({
  bots: initBots,
  sharedBots: initSharedBots,
}: {
  bots: Bot[];
  sharedBots: SharedBot[];
}) => {
  const [searchValue, setSearchValue] = useState<string>('');
  const page_bot = useTranslations('page_bot');

  const [bots, setBots] = useState<Bot[]>(initBots);

  const [sharedBots, setSharedBots] = useState<SharedBot[]>(initSharedBots);

  const getBots = useCallback(async () => {
    const res = await apiClient.defaultApi.botsGet();
    setBots(res.data.items || []);
  }, []);

  const getSharedBots = useCallback(async () => {
    const res = await apiClient.defaultApi.marketplaceBotsGet();
    setSharedBots(res.data.items || []);
  }, []);

  useInterval(() => {
    getBots();
    getSharedBots();
  }, 5000);

  return (
    <>
      <Tabs defaultValue="my_bots">
        <div className="flex flex-row items-center">
          <div className="flex flex-row gap-2">
            <TabsList>
              <TabsTrigger value="my_bots">{page_bot('my_bots')}</TabsTrigger>
              <TabsTrigger value="shared_bots">
                {page_bot('shared_bots')}
              </TabsTrigger>
            </TabsList>
            <Input
              placeholder={page_bot('search')}
              value={searchValue}
              onChange={(e) => setSearchValue(e.currentTarget.value)}
            />
          </div>
          <div className="ml-auto flex items-center gap-2">
            <Button asChild>
              <Link href="/bots/new">
                <Plus /> {page_bot('new_bot')}
              </Link>
            </Button>
          </div>
        </div>

        <TabsContent value="my_bots">
          <BotsCardList bots={bots} searchValue={searchValue} />
        </TabsContent>
        <TabsContent value="shared_bots">
          <BotsCardList bots={sharedBots} searchValue={searchValue} />
        </TabsContent>
      </Tabs>
    </>
  );
};
