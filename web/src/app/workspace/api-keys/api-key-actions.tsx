'use client';

import {
  createApiKey,
  deleteApiKey,
  updateApiKey,
} from '@/features/api-key/client-api';
import type { ApiKey } from '@/features/api-key/types';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormMessage,
} from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { zodResolver } from '@hookform/resolvers/zod';
import { Slot } from '@radix-ui/react-slot';
import copy from 'copy-to-clipboard';
import { Copy } from 'lucide-react';
import { useTranslations } from 'next-intl';
import { useRouter } from 'next/navigation';
import { useCallback, useState } from 'react';
import { useForm } from 'react-hook-form';
import { toast } from 'sonner';
import * as z from 'zod';

const apiKeySchema = z.object({
  description: z.string(),
});

export const ApiKeyActions = ({
  apiKey,
  action,
  children,
}: {
  apiKey?: ApiKey;
  action: 'add' | 'edit' | 'delete';
  children?: React.ReactNode;
}) => {
  const page_api_keys = useTranslations('page_api_keys');
  const common_action = useTranslations('common.action');
  const common_tips = useTranslations('common.tips');
  const components_copy_to_clipboard = useTranslations(
    'components.copy_to_clipboard',
  );
  const [createOrUpdateVisible, setCreateOrUpdateVisible] =
    useState<boolean>(false);
  const [deleteVisible, setDeleteVisible] = useState<boolean>(false);
  const [createdApiKey, setCreatedApiKey] = useState<ApiKey | null>(null);
  const [createdKeyVisible, setCreatedKeyVisible] = useState<boolean>(false);
  const router = useRouter();
  const form = useForm<z.infer<typeof apiKeySchema>>({
    resolver: zodResolver(apiKeySchema),
    defaultValues: {
      description: apiKey?.description || '',
    },
  });

  const handleCreateOrUpdate = useCallback(
    async (values: z.infer<typeof apiKeySchema>) => {
      if (action === 'edit' && apiKey?.id) {
        await updateApiKey(apiKey.id, values);
        setCreateOrUpdateVisible(false);
        setTimeout(router.refresh, 300);
        return;
      }
      if (action === 'add') {
        const created = await createApiKey(values);
        setCreateOrUpdateVisible(false);
        if (created) {
          setCreatedApiKey(created as ApiKey);
          setCreatedKeyVisible(true);
          form.reset({ description: '' });
          return;
        }
        setTimeout(router.refresh, 300);
      }
    },
    [action, apiKey?.id, form, router],
  );

  const handleDelete = useCallback(async () => {
    if (action === 'delete' && apiKey?.id) {
      await deleteApiKey(apiKey.id);
      setDeleteVisible(false);
      setTimeout(router.refresh, 300);
    }
  }, [action, apiKey?.id, router]);

  const handleCopyCreatedKey = useCallback(() => {
    if (!createdApiKey?.key) {
      return;
    }

    copy(createdApiKey.key);
    toast.success(components_copy_to_clipboard('copied'));
  }, [components_copy_to_clipboard, createdApiKey?.key]);

  const handleCloseCreatedKey = useCallback(() => {
    setCreatedKeyVisible(false);
    setCreatedApiKey(null);
    setTimeout(router.refresh, 300);
  }, [router]);

  if (action === 'delete') {
    return (
      <Dialog open={deleteVisible} onOpenChange={() => setDeleteVisible(false)}>
        <DialogTrigger asChild>
          <Slot
            onClick={(e) => {
              setDeleteVisible(true);
              e.preventDefault();
            }}
          >
            {children}
          </Slot>
        </DialogTrigger>
        <DialogContent showCloseButton={false}>
          <DialogHeader>
            <DialogTitle>{common_tips('confirm')}</DialogTitle>
            <DialogDescription>
              {page_api_keys('delete_api_key_confirm')}
            </DialogDescription>
          </DialogHeader>
          <DialogDescription></DialogDescription>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteVisible(false)}>
              {common_action('cancel')}
            </Button>
            <Button variant="destructive" onClick={() => handleDelete()}>
              {common_action('continue')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
  } else {
    return (
      <>
        <Dialog
          open={createOrUpdateVisible}
          onOpenChange={() => setCreateOrUpdateVisible(false)}
        >
          <DialogTrigger asChild>
            <Slot
              onClick={(e) => {
                setCreateOrUpdateVisible(true);
                e.preventDefault();
              }}
            >
              {children}
            </Slot>
          </DialogTrigger>
          <DialogContent showCloseButton={false}>
            <Form {...form}>
              <form
                onSubmit={form.handleSubmit(handleCreateOrUpdate)}
                className="space-y-8"
              >
                <DialogHeader>
                  <DialogTitle>API Key</DialogTitle>
                  <DialogDescription></DialogDescription>
                </DialogHeader>
                <FormField
                  control={form.control}
                  name="description"
                  render={({ field }) => (
                    <FormItem>
                      <FormControl>
                        <Textarea
                          placeholder={page_api_keys('api_key_placeholder')}
                          {...field}
                        />
                      </FormControl>
                      <FormDescription>
                        {page_api_keys('api_key_description')}
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <DialogFooter>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => setCreateOrUpdateVisible(false)}
                  >
                    {common_action('cancel')}
                  </Button>
                  <Button type="submit">{common_action('save')}</Button>
                </DialogFooter>
              </form>
            </Form>
          </DialogContent>
        </Dialog>
        <Dialog open={createdKeyVisible}>
          <DialogContent
            showCloseButton={false}
            onEscapeKeyDown={(event) => event.preventDefault()}
            onInteractOutside={(event) => event.preventDefault()}
          >
            <DialogHeader>
              <DialogTitle>{page_api_keys('created_api_key_title')}</DialogTitle>
              <DialogDescription>
                {page_api_keys('created_api_key_description')}
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-3">
              <div className="space-y-2">
                <p className="text-sm font-medium">
                  {page_api_keys('created_api_key_label')}
                </p>
                <Input
                  readOnly
                  value={createdApiKey?.key || ''}
                  className="font-mono text-xs"
                />
              </div>
              <p className="text-muted-foreground text-sm">
                {page_api_keys('created_api_key_hint')}
              </p>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={handleCopyCreatedKey}>
                <Copy />
                {page_api_keys('copy_created_api_key')}
              </Button>
              <Button onClick={handleCloseCreatedKey}>
                {page_api_keys('saved_api_key')}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </>
    );
  }
};
