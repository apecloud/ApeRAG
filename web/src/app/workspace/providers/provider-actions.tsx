'use client';

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog';
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
  FormLabel,
} from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import {
  createProvider,
  deleteProvider,
  publishProvider,
  updateProvider,
} from '@/features/providers/client-api';
import type { Provider } from '@/features/providers/types';
import { zodResolver } from '@hookform/resolvers/zod';
import { Slot } from '@radix-ui/react-slot';
import { useTranslations } from 'next-intl';
import { useRouter } from 'next/navigation';
import { useCallback, useState } from 'react';
import { type Resolver, useForm } from 'react-hook-form';
import { toast } from 'sonner';
import * as z from 'zod';

const defaultValue = {
  label: '',
  base_url: '',
  completion_dialect: 'openai',
  embedding_dialect: 'openai',
  rerank_dialect: 'jina_ai',
};

const providerSchema = z.object({
  label: z.string().min(1),
  base_url: z.string().min(1),
  completion_dialect: z.string().min(1),
  embedding_dialect: z.string().min(1),
  rerank_dialect: z.string().min(1),
});
type ProviderFormValues = z.infer<typeof providerSchema>;

export const ProviderActions = ({
  provider,
  action,
  children,
}: {
  provider?: Provider;
  action: 'add' | 'edit' | 'delete' | 'publish';
  children?: React.ReactNode;
}) => {
  const [createOrUpdateVisible, setCreateOrUpdateVisible] =
    useState<boolean>(false);
  const [deleteVisible, setDeleteVisible] = useState<boolean>(false);
  const [publishVisible, setPublishVisible] = useState<boolean>(false);
  const router = useRouter();

  const page_models = useTranslations('page_models');
  const common_action = useTranslations('common.action');
  const common_tips = useTranslations('common.tips');

  const form = useForm<ProviderFormValues>({
    resolver: zodResolver(providerSchema) as Resolver<ProviderFormValues>,
    defaultValues: {
      label: provider?.label ?? defaultValue.label,
      base_url: provider?.base_url ?? defaultValue.base_url,
      completion_dialect:
        provider?.completion_dialect ?? defaultValue.completion_dialect,
      embedding_dialect:
        provider?.embedding_dialect ?? defaultValue.embedding_dialect,
      rerank_dialect: provider?.rerank_dialect ?? defaultValue.rerank_dialect,
    },
  });

  const handleDelete = useCallback(async () => {
    if (action === 'delete' && provider?.name) {
      await deleteProvider(provider.name);
      setDeleteVisible(false);
      setTimeout(router.refresh, 300);
    }
  }, [action, provider?.name, router]);

  const handlePublish = useCallback(async () => {
    if (action === 'publish' && provider?.name) {
      await publishProvider(provider.name);
      setPublishVisible(false);
      toast.success(page_models('provider.publish_success'));
      setTimeout(router.refresh, 300);
    }
  }, [action, provider?.name, router, page_models]);

  const handleCreateOrUpdate = useCallback(
    async (values: ProviderFormValues) => {
      let res;
      const { data: params, error } = providerSchema.safeParse(values);
      if (error) {
        return;
      }

      if (action === 'edit' && provider?.name) {
        res = await updateProvider(provider.name, params);
      }
      if (action === 'add') {
        res = await createProvider(params);
      }
      if (res) {
        setCreateOrUpdateVisible(false);
        setTimeout(router.refresh, 300);
        toast.success(common_tips('save_success'));
      }
    },
    [action, common_tips, provider?.name, router.refresh],
  );

  if (action === 'delete') {
    return (
      <AlertDialog
        open={deleteVisible}
        onOpenChange={() => setDeleteVisible(false)}
      >
        <AlertDialogTrigger asChild>
          <Slot
            onClick={(e) => {
              setDeleteVisible(true);
              e.preventDefault();
            }}
          >
            {children}
          </Slot>
        </AlertDialogTrigger>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{common_tips('confirm')}</AlertDialogTitle>
            <AlertDialogDescription>
              {page_models('provider.delete_confirm')}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogDescription></AlertDialogDescription>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setDeleteVisible(false)}>
              {common_action('cancel')}
            </AlertDialogCancel>
            <AlertDialogAction onClick={() => handleDelete()}>
              {common_action('continue')}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    );
  } else if (action === 'publish') {
    return (
      <AlertDialog
        open={publishVisible}
        onOpenChange={() => setPublishVisible(false)}
      >
        <AlertDialogTrigger asChild>
          <Slot
            onClick={(e) => {
              setPublishVisible(true);
              e.preventDefault();
            }}
          >
            {children}
          </Slot>
        </AlertDialogTrigger>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{common_tips('confirm')}</AlertDialogTitle>
            <AlertDialogDescription>
              {page_models('provider.publish_confirm')}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setPublishVisible(false)}>
              {common_action('cancel')}
            </AlertDialogCancel>
            <AlertDialogAction onClick={() => handlePublish()}>
              {common_action('continue')}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    );
  } else {
    return (
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
        <DialogContent>
          <Form {...form}>
            <form
              onSubmit={form.handleSubmit(handleCreateOrUpdate)}
              className="space-y-8"
            >
              <DialogHeader>
                <DialogTitle>
                  {action === 'add' && page_models('provider.add_provider')}
                  {action === 'edit' && page_models('provider.edit_provider')}
                </DialogTitle>
                <DialogDescription></DialogDescription>
              </DialogHeader>
              <FormField
                control={form.control}
                name="label"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>{page_models('provider.name')}</FormLabel>
                    <FormControl>
                      <Input
                        placeholder={page_models('provider.name_placeholder')}
                        {...field}
                      />
                    </FormControl>
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="base_url"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>{page_models('provider.base_url')}</FormLabel>
                    <FormControl>
                      <Input
                        placeholder={page_models(
                          'provider.base_url_placeholder',
                        )}
                        {...field}
                      />
                    </FormControl>
                    <FormDescription>
                      {page_models('provider.base_url_description')}
                    </FormDescription>
                  </FormItem>
                )}
              />
              <div>
                <FormLabel className="text-muted-foreground mb-4">
                  {page_models('provider.api_dialect')}
                </FormLabel>
                <div className="grid grid-cols-3 gap-4">
                  <FormField
                    control={form.control}
                    name="completion_dialect"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Completion</FormLabel>
                        <FormControl>
                          <Input
                            placeholder="Completion API Dialect"
                            {...field}
                          />
                        </FormControl>
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="embedding_dialect"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Embedding</FormLabel>
                        <FormControl>
                          <Input
                            placeholder="Embedding API Dialect"
                            {...field}
                          />
                        </FormControl>
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="rerank_dialect"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Rerank</FormLabel>
                        <FormControl>
                          <Input
                            placeholder="Rerank API Dialect
"
                            {...field}
                          />
                        </FormControl>
                      </FormItem>
                    )}
                  />
                </div>
                <div className="text-muted-foreground mt-2 text-sm">
                  {page_models('provider.api_dialect_description')}
                </div>
              </div>

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
    );
  }
};
