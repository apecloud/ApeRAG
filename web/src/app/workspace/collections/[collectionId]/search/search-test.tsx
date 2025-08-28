import { SearchRequest } from '@/api';
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
import { apiClient } from '@/lib/api/client';
import { zodResolver } from '@hookform/resolvers/zod';
import { Slot } from '@radix-ui/react-slot';
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
  summary_search: z
    .object({
      topk: z.number(),
      similarity: z.number(),
    })
    .optional(),
});

type IndexType = 'vector' | 'fulltext' | 'graph' | 'summary';

export const SearchTest = ({ children }: { children: ReactNode }) => {
  const [visible, setVisible] = useState<boolean>(false);
  const { collection } = useCollectionContext();
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
    summary: {
      available: Boolean(collection.config?.enable_summary),
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
      summary_search: {
        topk: 5,
        similarity: 0.7,
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
        rerank: true,
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
      if (indexTypes.summary.checked) {
        data.summary_search = values.summary_search;
      }

      await apiClient.defaultApi.collectionsCollectionIdSearchesPost({
        collectionId: collection.id,
        searchRequest: data,
      });

      setVisible(false);

      router.refresh();
    },
    [
      collection.id,
      indexTypes.fulltext.checked,
      indexTypes.graph.checked,
      indexTypes.summary.checked,
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
      <DialogContent>
        <Form {...form}>
          <form
            onSubmit={form.handleSubmit(handleSubmit)}
            className="space-y-6"
          >
            <DialogHeader>
              <DialogTitle>Search test</DialogTitle>
              <DialogDescription></DialogDescription>
            </DialogHeader>

            <div className="flex flex-col gap-4">
              <FormField
                control={form.control}
                name="query"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Query</FormLabel>
                    <FormControl>
                      <Textarea placeholder="Enter your question." {...field} />
                    </FormControl>
                  </FormItem>
                )}
              />

              {indexTypes.vector.available && (
                <div className="bg-accent/40 hover:bg-accent/50 flex flex-row gap-4 rounded-md border p-4">
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
                    Vector
                  </Label>

                  <div className="flex w-[65%] flex-col gap-2">
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
                <div className="bg-accent/40 hover:bg-accent/50 flex flex-row gap-4 rounded-md border p-4">
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
                    Fulltext
                  </Label>

                  <div className="flex w-[65%] flex-col gap-2">
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
                <div className="bg-accent/40 hover:bg-accent/50 flex flex-row gap-4 rounded-md border p-4">
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
                    Graph
                  </Label>
                  <div className="flex w-[65%] flex-col gap-2">
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

              {indexTypes.summary.available && (
                <div className="bg-accent/40 hover:bg-accent/50 flex flex-row gap-4 rounded-md border p-4">
                  <Label className="h-8 flex-1">
                    <Checkbox
                      checked={indexTypes.summary.checked}
                      onCheckedChange={(checked) => {
                        setIndexTypes((data) => {
                          data.summary.checked = Boolean(checked);
                          return { ...data };
                        });
                      }}
                    />
                    Summary
                  </Label>
                  <div className="flex w-[65%] flex-col gap-2">
                    <FormField
                      control={form.control}
                      name="summary_search.topk"
                      render={({ field }) => (
                        <FormItem className="gap-1">
                          <FormControl>
                            <Slider
                              value={[field.value]}
                              onValueChange={(v) => field.onChange(v[0])}
                              min={0}
                              max={20}
                              step={1}
                              disabled={!indexTypes.summary.checked}
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
                      name="summary_search.similarity"
                      render={({ field }) => (
                        <FormItem className="gap-1">
                          <FormControl>
                            <Slider
                              value={[field.value]}
                              onValueChange={(v) => field.onChange(v[0])}
                              min={0}
                              max={1}
                              step={0.1}
                              disabled={!indexTypes.summary.checked}
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
            </div>
            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setVisible(false)}
              >
                Cancel
              </Button>
              <Button type="submit">Test</Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
};
