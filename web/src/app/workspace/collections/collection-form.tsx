'use client';

import { Collection, ModelSpec } from '@/api';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
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
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import { apiClient } from '@/lib/api/client';
import { cn, objectKeys } from '@/lib/utils';
import { zodResolver } from '@hookform/resolvers/zod';
import _ from 'lodash';
import { useRouter } from 'next/navigation';
import { useCallback, useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { toast } from 'sonner';
import * as z from 'zod';
import { CollectionConfigIndexTypes } from './tools';

const collectionModelSchema = z
  .object({
    custom_llm_provider: z.string(),
    model: z.string(),
    model_service_provider: z.string(),
  })
  .optional();

const collectionSchema = z
  .object({
    title: z.string().min(1),
    description: z.string(),
    type: z.enum(['document']),
    config: z.object({
      source: z.enum(['system']),
      enable_fulltext: z.boolean(),
      enable_knowledge_graph: z.boolean(),
      enable_summary: z.boolean(),
      enable_vector: z.boolean(),
      enable_vision: z.boolean(),
      completion: collectionModelSchema,
      embedding: collectionModelSchema,
    }),
  })
  .refine(
    ({ config }) => {
      if (config.enable_vector) {
        return !_.isEmpty(config.embedding?.model);
      }
      return true;
    },
    {
      path: ['config.embedding.model'],
    },
  )
  .refine(
    ({ config }) => {
      if (
        config.enable_knowledge_graph ||
        config.enable_summary ||
        config.enable_vision
      ) {
        return !_.isEmpty(config.completion?.model);
      }
      return true;
    },
    {
      path: ['config.completion.model'],
    },
  );

const defaultValues: z.infer<typeof collectionSchema> = {
  title: '',
  description: '',
  type: 'document',
  config: {
    source: 'system',
    enable_fulltext: true,
    enable_knowledge_graph: true,
    enable_vector: true,
    enable_summary: false,
    enable_vision: false,
    completion: {
      custom_llm_provider: '',
      model: '',
      model_service_provider: '',
    },
    embedding: {
      custom_llm_provider: '',
      model: '',
      model_service_provider: '',
    },
  },
};

export type ProviderModels = {
  label?: string;
  name?: string;
  models?: ModelSpec[];
}[];

export const CollectionForm = ({
  collection,
  action,
}: {
  collection?: Collection;
  action: 'add' | 'edit';
}) => {
  const router = useRouter();
  const [completionModels, setCompletionModels] = useState<ProviderModels>();
  const [embeddingModels, setEmbeddingModels] = useState<ProviderModels>();

  const form = useForm<z.infer<typeof collectionSchema>>({
    resolver: zodResolver(collectionSchema),
    defaultValues: {
      ...defaultValues,
    },
  });

  const loadModels = useCallback(async () => {
    const res = await apiClient.defaultApi.availableModelsPost({
      tagFilterRequest: {
        tag_filters: [{ operation: 'AND', tags: ['enable_for_collection'] }],
      },
    });
    const completion = res.data.items?.map((m) => {
      return {
        label: m.label,
        name: m.name,
        models: m.completion,
      };
    });
    const embedding = res.data.items?.map((m) => {
      return {
        label: m.label,
        name: m.name,
        models: m.embedding,
      };
    });
    setCompletionModels(completion || []);
    setEmbeddingModels(embedding || []);
  }, []);

  const handleCreateOrUpdate = useCallback(
    async (values: z.infer<typeof collectionSchema>) => {
      if (action === 'edit') {
        if (!collection?.id) return;
        const res = await apiClient.defaultApi.collectionsCollectionIdPut({
          collectionId: collection.id,
          collectionUpdate: values,
        });
        if (res.data.id) {
          toast.success('Saved successfully.');
        }
      }
      if (action === 'add') {
        const res = await apiClient.defaultApi.collectionsPost({
          collectionCreate: values,
        });
        if (res.data.id) {
          toast.success('Saved successfully.');
          router.push('/workspace/collections');
        }
      }
    },
    [action, collection?.id, router],
  );

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  return (
    <>
      <Form {...form}>
        <form
          onSubmit={form.handleSubmit(handleCreateOrUpdate)}
          className="flex flex-col gap-4"
        >
          <Card>
            <CardHeader>
              <CardTitle>General</CardTitle>
              <CardDescription></CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-6">
              <FormField
                control={form.control}
                name="title"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Name</FormLabel>
                    <FormControl>
                      <Input
                        className="md:w-6/12"
                        placeholder="Collection display name."
                        {...field}
                      />
                    </FormControl>
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="description"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Description</FormLabel>
                    <FormControl>
                      <Textarea
                        className="h-25"
                        placeholder="Please describe the general meaning of a collection."
                        {...field}
                      />
                    </FormControl>
                  </FormItem>
                )}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Index Types</CardTitle>
              <CardDescription>
                Select the AI capabilities you need, we will build corresponding
                indexes for your documents
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              {objectKeys(CollectionConfigIndexTypes).map((key) => {
                const item = CollectionConfigIndexTypes[key];
                return (
                  <FormField
                    key={key}
                    control={form.control}
                    name={key}
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel
                          className={cn(
                            'has-[[aria-checked=true]]:bg-accent/50 flex items-center gap-3 rounded-lg border p-3',
                            item.disabled
                              ? 'cursor-not-allowed'
                              : 'hover:bg-accent/30 cursor-pointer',
                          )}
                        >
                          <div className="grid gap-2">
                            <div className="flex items-center gap-2 leading-none font-medium">
                              {item.title}
                              {item.disabled && <Badge>Required</Badge>}
                            </div>
                            <p className="text-muted-foreground text-sm">
                              {item.description}
                            </p>
                          </div>
                          <FormControl className="ml-auto">
                            <Switch
                              checked={field.value}
                              disabled={item.disabled}
                              onCheckedChange={field.onChange}
                            />
                          </FormControl>
                        </FormLabel>
                      </FormItem>
                    )}
                  />
                );
              })}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Model Settings</CardTitle>
              <CardDescription>
                Select AI models for document processing. Different index types
                require different model support
              </CardDescription>
            </CardHeader>

            <CardContent className="flex flex-col gap-6 pt-6">
              <FormField
                control={form.control}
                name="config.embedding.model"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Embedding Model</FormLabel>
                    <FormControl className="ml-auto">
                      <Select {...field} onValueChange={field.onChange}>
                        <SelectTrigger className="w-full cursor-pointer md:w-6/12">
                          <SelectValue placeholder="Select a model" />
                        </SelectTrigger>
                        <SelectContent>
                          {embeddingModels
                            ?.filter((item) => _.size(item.models))
                            .map((item) => {
                              return (
                                <SelectGroup key={item.name}>
                                  <SelectLabel>{item.label}</SelectLabel>
                                  {item.models?.map((model) => {
                                    return (
                                      <SelectItem
                                        key={model.model}
                                        value={model.model || ''}
                                      >
                                        {model.model}
                                      </SelectItem>
                                    );
                                  })}
                                </SelectGroup>
                              );
                            })}
                        </SelectContent>
                      </Select>
                    </FormControl>
                    <FormDescription>
                      An embedding model translates data into numerical vectors
                      that capture their semantic meaning and relationships.
                    </FormDescription>
                  </FormItem>
                )}
              />

              <Separator />

              <FormField
                control={form.control}
                name="config.completion.model"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Completion Model</FormLabel>
                    <FormControl className="ml-auto">
                      <Select {...field} onValueChange={field.onChange}>
                        <SelectTrigger className="w-full cursor-pointer md:w-6/12">
                          <SelectValue placeholder="Select a model" />
                        </SelectTrigger>
                        <SelectContent>
                          {completionModels
                            ?.filter((item) => _.size(item.models))
                            .map((item) => {
                              return (
                                <SelectGroup key={item.name}>
                                  <SelectLabel>{item.label}</SelectLabel>
                                  {item.models?.map((model) => {
                                    return (
                                      <SelectItem
                                        key={model.model}
                                        value={model.model || ''}
                                      >
                                        {model.model}
                                      </SelectItem>
                                    );
                                  })}
                                </SelectGroup>
                              );
                            })}
                        </SelectContent>
                      </Select>
                    </FormControl>
                    <FormDescription>
                      A completion model is an AI that generates new content by
                      predicting the most likely subsequent text based on a
                      given input.
                    </FormDescription>
                  </FormItem>
                )}
              />
            </CardContent>
          </Card>

          <div className="flex justify-end">
            <Button type="submit" className="px-12">
              Save
            </Button>
          </div>
        </form>
      </Form>
    </>
  );
};
