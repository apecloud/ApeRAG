'use client';

import { LlmProvider } from '@/api';
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
  FormField,
  FormItem,
  FormLabel,
} from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { apiClient } from '@/lib/api/client';
import { zodResolver } from '@hookform/resolvers/zod';
import { Slot } from '@radix-ui/react-slot';
import { useRouter } from 'next/navigation';
import { useCallback, useState } from 'react';
import { useForm } from 'react-hook-form';
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

export const ProviderActions = ({
  provider,
  action,
  children,
}: {
  provider?: LlmProvider;
  action: 'add' | 'edit' | 'delete';
  children?: React.ReactNode;
}) => {
  const [createOrUpdateVisible, setCreateOrUpdateVisible] =
    useState<boolean>(false);
  const [deleteVisible, setDeleteVisible] = useState<boolean>(false);
  const router = useRouter();

  const form = useForm<z.infer<typeof providerSchema>>({
    resolver: zodResolver(providerSchema),
    defaultValues: { ...defaultValue, ...provider },
  });

  const handleDelete = useCallback(async () => {
    if (action === 'delete' && provider?.name) {
      const res = await apiClient.defaultApi.llmProvidersProviderNameDelete({
        providerName: provider.name,
      });
      if (res?.status === 200) {
        setDeleteVisible(false);
        setTimeout(router.refresh, 300);
      }
    }
  }, [action, provider?.name, router]);

  const handleCreateOrUpdate = useCallback(
    async (values: z.infer<typeof providerSchema>) => {
      let res;
      const { data: params, error } = providerSchema.safeParse(values);
      if (error) {
        return;
      }

      if (action === 'edit' && provider?.name) {
        res = await apiClient.defaultApi.llmProvidersProviderNamePut({
          providerName: provider.name,
          llmProviderUpdateWithApiKey: params,
        });
      }
      if (action === 'add') {
        res = await apiClient.defaultApi.llmProvidersPost({
          llmProviderCreateWithApiKey: params,
        });
      }
      if (res?.status === 200) {
        setCreateOrUpdateVisible(false);
        setTimeout(router.refresh, 300);
        toast.success('Saved successfully.');
      }
    },
    [action, provider?.name, router],
  );

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
            <DialogTitle>Confirm</DialogTitle>
            <DialogDescription>
              Confirm deletion of the provider?
            </DialogDescription>
          </DialogHeader>
          <DialogDescription></DialogDescription>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteVisible(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={() => handleDelete()}>
              OK
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
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
                <DialogTitle>Provider</DialogTitle>
                <DialogDescription></DialogDescription>
              </DialogHeader>
              <FormField
                control={form.control}
                name="label"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Name</FormLabel>
                    <FormControl>
                      <Input placeholder="Provider display name." {...field} />
                    </FormControl>
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="base_url"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>API base url</FormLabel>
                    <FormControl>
                      <Input placeholder="API base url" {...field} />
                    </FormControl>
                  </FormItem>
                )}
              />
              <div>
                <FormLabel className="text-muted-foreground mb-4">
                  API Dialect
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
              </div>

              <DialogFooter>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setCreateOrUpdateVisible(false)}
                >
                  Cancel
                </Button>
                <Button type="submit">Save</Button>
              </DialogFooter>
            </form>
          </Form>
        </DialogContent>
      </Dialog>
    );
  }
};
