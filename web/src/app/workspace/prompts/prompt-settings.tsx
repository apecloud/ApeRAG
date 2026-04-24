'use client';

import {
  deleteUserPrompt,
  updateUserPrompts,
} from '@/features/prompt/client-api';
import type {
  PromptDetail,
  UserPromptsResponse,
} from '@/features/prompt/types';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Textarea } from '@/components/ui/textarea';
import { cn } from '@/lib/utils';
import { MessageSquareText, ShieldCheck } from 'lucide-react';
import { useTranslations } from 'next-intl';
import { useRouter } from 'next/navigation';
import { type ElementType, useCallback, useEffect, useState } from 'react';
import { toast } from 'sonner';

type PromptType = 'agent_system' | 'agent_query';

const PROMPT_TYPES: PromptType[] = ['agent_system', 'agent_query'];
const PROMPT_ICON: Record<PromptType, ElementType> = {
  agent_system: ShieldCheck,
  agent_query: MessageSquareText,
};

interface PromptCardProps {
  promptType: PromptType;
  detail: PromptDetail | undefined;
  onSaved: () => void;
}

const PromptCard = ({ promptType, detail, onSaved }: PromptCardProps) => {
  const page_prompts = useTranslations('page_prompts');
  const common_action = useTranslations('common.action');
  const [content, setContent] = useState(detail?.content ?? '');
  const [saving, setSaving] = useState(false);
  const [resetOpen, setResetOpen] = useState(false);

  useEffect(() => {
    setContent(detail?.content ?? '');
  }, [detail?.content]);

  const handleSave = useCallback(async () => {
    setSaving(true);
    try {
      await updateUserPrompts({
        prompts: { [promptType]: content },
      });
      toast.success(page_prompts('toast.save_success'));
      onSaved();
    } finally {
      setSaving(false);
    }
  }, [content, promptType, page_prompts, onSaved]);

  const handleReset = useCallback(async () => {
    await deleteUserPrompt(promptType);
    toast.success(page_prompts('toast.reset_success'));
    setResetOpen(false);
    onSaved();
  }, [promptType, page_prompts, onSaved]);

  const isCustomized = detail?.customized === true;
  const Icon = PROMPT_ICON[promptType];

  return (
    <Card className="gap-0 overflow-hidden rounded-xl border-border/70 py-0">
      <CardHeader className="border-b border-border/70 px-5 py-5">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="flex min-w-0 items-center gap-3">
            <div className="bg-accent-soft text-accent-ink flex size-10 shrink-0 items-center justify-center rounded-lg">
              <Icon className="size-5" />
            </div>
            <div className="min-w-0">
              <CardTitle className="font-serif text-xl font-normal">
                {page_prompts(`${promptType}.title` as never)}
              </CardTitle>
              <CardDescription className="mt-1 leading-6">
                {page_prompts(`${promptType}.description` as never)}
              </CardDescription>
            </div>
          </div>
          <Badge
            variant="outline"
            className={cn(
              'rounded-full',
              isCustomized
                ? 'border-primary/20 bg-accent-soft text-accent-ink'
                : 'bg-muted text-muted-foreground',
            )}
          >
            {isCustomized
              ? page_prompts('status.customized')
              : page_prompts('status.default')}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="bg-muted/60 px-5 py-5">
        <Textarea
          className="bg-card min-h-[220px] max-h-[460px] resize-y rounded-xl border-border/70 font-mono text-sm leading-6"
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder={detail?.content ?? ''}
        />
      </CardContent>

      <CardFooter className="justify-end gap-2 border-t border-border/70 px-5 py-4">
        {isCustomized && (
          <Dialog open={resetOpen} onOpenChange={setResetOpen}>
            <DialogTrigger asChild>
              <Button variant="outline">
                {page_prompts('action.reset')}
              </Button>
            </DialogTrigger>
            <DialogContent className="rounded-xl border-border/70">
              <DialogHeader>
                <DialogTitle className="font-serif text-2xl font-normal">
                  {page_prompts('action.reset_confirm')}
                </DialogTitle>
                <DialogDescription>
                  {page_prompts('action.reset_confirm_description')}
                </DialogDescription>
              </DialogHeader>
              <DialogFooter>
                <DialogClose asChild>
                  <Button variant="outline">{common_action('cancel')}</Button>
                </DialogClose>
                <Button variant="destructive" onClick={handleReset}>
                  {page_prompts('action.reset')}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        )}
        <Button onClick={handleSave} disabled={saving}>
          {page_prompts('action.save')}
        </Button>
      </CardFooter>
    </Card>
  );
};

export const PromptSettings = ({ data }: { data: UserPromptsResponse }) => {
  const router = useRouter();

  const handleSaved = useCallback(() => {
    setTimeout(router.refresh, 300);
  }, [router]);

  return (
    <div className="grid gap-5 xl:grid-cols-2">
      {PROMPT_TYPES.map((promptType) => (
        <PromptCard
          key={promptType}
          promptType={promptType}
          detail={data[promptType]}
          onSaved={handleSaved}
        />
      ))}
    </div>
  );
};
