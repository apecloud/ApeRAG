import Link from 'next/link';

import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { FlaskConical, MessageSquareText } from 'lucide-react';
import { getTranslations } from 'next-intl/server';

export const BotSectionNav = async ({
  botId,
  active,
}: {
  botId: string;
  active: 'chats' | 'evaluation';
}) => {
  const pageChat = await getTranslations('page_chat');
  const pageBotEvaluation = (await getTranslations(
    'page_bot_evaluation' as never,
  )) as (key: 'metadata.title') => string;

  const links = [
    {
      key: 'chats',
      href: `/workspace/bots/${botId}/chats`,
      label: pageChat('metadata.title'),
      icon: MessageSquareText,
    },
    {
      key: 'evaluation',
      href: `/workspace/bots/${botId}/evaluation`,
      label: pageBotEvaluation('metadata.title'),
      icon: FlaskConical,
    },
  ] as const;

  return (
    <div className="flex flex-wrap items-center gap-2">
      {links.map((link) => {
        const Icon = link.icon;
        const isActive = active === link.key;
        return (
          <Button
            key={link.key}
            asChild
            variant={isActive ? 'default' : 'outline'}
            className={cn('rounded-full', !isActive && 'bg-background')}
          >
            <Link href={link.href}>
              <Icon className="size-4" />
              {link.label}
            </Link>
          </Button>
        );
      })}
    </div>
  );
};
