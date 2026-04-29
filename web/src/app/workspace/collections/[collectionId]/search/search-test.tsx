import { useCollectionContext } from '@/components/providers/collection-provider';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
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
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Textarea } from '@/components/ui/textarea';
import { createSearch } from '@/features/retrieval/client-api';
import type { SearchRequest } from '@/features/retrieval/types';
import { zodResolver } from '@hookform/resolvers/zod';
import { Slot } from '@radix-ui/react-slot';
import { useTranslations } from 'next-intl';
import { useRouter } from 'next/navigation';
import { ReactNode, useCallback, useState } from 'react';
import { useForm } from 'react-hook-form';

import * as z from 'zod';

const searchParams = z.object({
  query: z.string().min(1),
  vector_search: z
    .object({
      topk: z.number(),
      similarity: z.number(),
    })
    .optional(),
  fulltext_search: z
    .object({
      topk: z.number(),
      // keywords: z.array(z.string())
    })
    .optional(),
  graph_search: z
    .object({
      topk: z.number(),
    })
    .optional(),
});

type IndexType = 'vector' | 'fulltext' | 'graph';

export const SearchTest = ({ children }: { children: ReactNode }) => {
  const [visible, setVisible] = useState<boolean>(false);
  const { collection } = useCollectionContext();
  const page_search = useTranslations('page_search');
  const common_action = useTranslations('common.action');
  const router = useRouter();
  const [indexTypes, setIndexTypes] = useState<{
    [key in IndexType]: {
      available: boolean;
      checked: boolean;
    };
  }>({
    vector: {
      available: Boolean(collection.config?.enable_vector),
      checked: true,
    },
    fulltext: {
      available: Boolean(collection.config?.enable_fulltext),
      checked: true,
    },
    graph: {
      available: Boolean(collection.config?.enable_knowledge_graph),
      checked: false,
    },
  });

  const form = useForm<z.infer<typeof searchParams>>({
    resolver: zodResolver(searchParams),
    defaultValues: {
      query: '',
      vector_search: {
        topk: 5,
        similarity: 0.7,
      },
      fulltext_search: {
        topk: 5,
      },
      graph_search: {
        topk: 5,
      },
    },
  });

  const handleSubmit = useCallback(
    async (values: z.infer<typeof searchParams>) => {
      if (!collection.id) {
        return;
      }

      const data: SearchRequest = {
        query: values.query,
        save_to_history: true,
      };

      if (indexTypes.fulltext.checked) {
        data.fulltext_search = values.fulltext_search;
      }
      if (indexTypes.vector.checked) {
        data.vector_search = values.vector_search;
      }
      if (indexTypes.graph.checked) {
        data.graph_search = values.graph_search;
      }

      await createSearch(collection.id, data);

      setVisible(false);

      router.refresh();
    },
    [
      collection.id,
      indexTypes.fulltext.checked,
      indexTypes.graph.checked,
      indexTypes.vector.checked,
      router,
    ],
  );

  return (
    <Dialog open={visible} onOpenChange={() => setVisible(false)}>
      <DialogTrigger asChild>
        <Slot
          onClick={(e) => {
            setVisible(true);
            e.preventDefault();
          }}
        >
          {children}
        </Slot>
      </DialogTrigger>
      <DialogContent className="sm:max-w-2xl">
        <Form {...form}>
          <form
            onSubmit={form.handleSubmit(handleSubmit)}
            className="space-y-6"
          >
            <DialogHeader>
              <DialogTitle>{page_search('metadata.title')}</DialogTitle>
              <DialogDescription>
                {page_search('questions_placeholder')}
              </DialogDescription>
            </DialogHeader>

            <div className="flex flex-col gap-4">
              <FormField
                control={form.control}
                name="query"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>{page_search('questions')}</FormLabel>
                    <FormControl>
                      <Textarea
                        placeholder={page_search('questions_placeholder')}
                        {...field}
                      />
                    </FormControl>
                  </FormItem>
                )}
              />

              {indexTypes.vector.available && (
                <div className="border-border/70 bg-muted/40 flex flex-col gap-4 rounded-xl border p-4 sm:flex-row">
                  <Label className="h-8 flex-1">
                    <Checkbox
                      checked={indexTypes.vector.checked}
                      onCheckedChange={(checked) => {
                        setIndexTypes((data) => {
                          data.vector.checked = Boolean(checked);
                          return { ...data };
                        });
                      }}
                    />
                    {page_search('vector_search')}
                  </Label>

                  <div className="flex flex-1 flex-col gap-2 sm:w-[65%]">
                    <FormField
                      control={form.control}
                      name="vector_search.topk"
                      render={({ field }) => (
                        <FormItem className="gap-1">
                          <FormControl>
                            <Slider
                              value={[field.value]}
                              onValueChange={(v) => field.onChange(v[0])}
                              min={0}
                              max={20}
                              step={1}
                              disabled={!indexTypes.vector.checked}
                            />
                          </FormControl>
                          <div className="text-muted-foreground flex flex-row justify-between text-xs">
                            <div>topK</div>
                            <div>{field.value}</div>
                          </div>
                        </FormItem>
                      )}
                    />
                    <FormField
                      control={form.control}
                      name="vector_search.similarity"
                      render={({ field }) => (
                        <FormItem className="gap-1">
                          <FormControl>
                            <Slider
                              value={[field.value]}
                              onValueChange={(v) => field.onChange(v[0])}
                              min={0}
                              max={1}
                              step={0.1}
                              disabled={!indexTypes.vector.checked}
                            />
                          </FormControl>
                          <div className="text-muted-foreground flex flex-row justify-between text-xs">
                            <div>Similarity</div>
                            <div>{field.value}</div>
                          </div>
                        </FormItem>
                      )}
                    />
                  </div>
                </div>
              )}

              {indexTypes.fulltext.available && (
                <div className="border-border/70 bg-muted/40 flex flex-col gap-4 rounded-xl border p-4 sm:flex-row">
                  <Label className="h-8 flex-1">
                    <Checkbox
                      checked={indexTypes.fulltext.checked}
                      onCheckedChange={(checked) => {
                        setIndexTypes((data) => {
                          data.fulltext.checked = Boolean(checked);
                          return { ...data };
                        });
                      }}
                    />
                    {page_search('fulltext_search')}
                  </Label>

                  <div className="flex flex-1 flex-col gap-2 sm:w-[65%]">
                    <FormField
                      control={form.control}
                      name="fulltext_search.topk"
                      render={({ field }) => (
                        <FormItem className="gap-1">
                          <FormControl>
                            <Slider
                              value={[field.value]}
                              onValueChange={(v) => field.onChange(v[0])}
                              min={0}
                              max={20}
                              step={1}
                              disabled={!indexTypes.fulltext.checked}
                            />
                          </FormControl>
                          <div className="text-muted-foreground flex flex-row justify-between text-xs">
                            <div>topK</div>
                            <div>{field.value}</div>
                          </div>
                        </FormItem>
                      )}
                    />
                  </div>
                </div>
              )}

              {indexTypes.graph.available && (
                <div className="border-border/70 bg-muted/40 flex flex-col gap-4 rounded-xl border p-4 sm:flex-row">
                  <Label className="h-8 flex-1">
                    <Checkbox
                      checked={indexTypes.graph.checked}
                      onCheckedChange={(checked) => {
                        setIndexTypes((data) => {
                          data.graph.checked = Boolean(checked);
                          return { ...data };
                        });
                      }}
                    />
                    {page_search('graph_search')}
                  </Label>
                  <div className="flex flex-1 flex-col gap-2 sm:w-[65%]">
                    <FormField
                      control={form.control}
                      name="graph_search.topk"
                      render={({ field }) => (
                        <FormItem className="gap-1">
                          <FormControl>
                            <Slider
                              value={[field.value]}
                              onValueChange={(v) => field.onChange(v[0])}
                              min={0}
                              max={20}
                              step={1}
                              disabled={!indexTypes.graph.checked}
                            />
                          </FormControl>
                          <div className="text-muted-foreground flex flex-row justify-between text-xs">
                            <div>topK</div>
                            <div>{field.value}</div>
                          </div>
                        </FormItem>
                      )}
                    />
                  </div>
                </div>
              )}
            </div>
            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setVisible(false)}
              >
                {common_action('cancel')}
              </Button>
              <Button type="submit">{common_action('continue')}</Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
};
