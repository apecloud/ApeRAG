'use client';

import {
  TITLE_LANGUAGES,
  type TitleLanguage,
} from '@/features/collection/types';
import type { ModelSpec } from '@/features/providers/types';
import { useCollectionContext } from '@/components/providers/collection-provider';
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
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
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
import {
  createCollection,
  updateCollection,
} from '@/features/collection/client-api';
import { getAvailableModels } from '@/features/providers/client-api';
import { cn, objectKeys } from '@/lib/utils';
import { zodResolver } from '@hookform/resolvers/zod';
import { ArrowLeft, Database, Sparkles } from 'lucide-react';
import _ from 'lodash';
import { useLocale, useTranslations } from 'next-intl';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useCallback, useEffect, useState } from 'react';
import { useForm, useWatch } from 'react-hook-form';
import { toast } from 'sonner';
import * as z from 'zod';
import { isVisibleCollectionConfigKey } from './feature-visibility';

const modelSelectLabel = (m: ModelSpec) =>
  (m.display_name?.trim() || m.model_id || '').trim();

const collectionModelSchema = z
  .object({
    model_id: z.string(),
    temperature: z.number().nullable(),
    tags: z.array(z.string()).nullable(),
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
      graph_backend_type: z.enum(['postgres', 'neo4j', 'nebula']).nullable(),
      fulltext_backend_type: z.enum(['elasticsearch', 'opensearch']).nullable(),
      completion: collectionModelSchema,
      embedding: collectionModelSchema,
      language: z.enum(TITLE_LANGUAGES),
    }),
  })
  .refine(
    ({ config }) => {
      if (config.enable_vector) {
        return !_.isEmpty(config.embedding?.model_id);
      }
      return true;
    },
    {
      path: ['config.embedding.model_id'],
    },
  )
  .refine(
    ({ config }) => {
      if (
        config.enable_knowledge_graph ||
        config.enable_summary ||
        config.enable_vision
      ) {
        return !_.isEmpty(config.completion?.model_id);
      }
      return true;
    },
    {
      path: ['config.completion.model_id'],
    },
  );

type FormValueType = z.infer<typeof collectionSchema>;

export type ProviderModel = {
  label?: string;
  name?: string;
  models?: ModelSpec[];
};

export const CollectionForm = ({ action }: { action: 'add' | 'edit' }) => {
  const router = useRouter();
  const { collection, loadCollection } = useCollectionContext();
  const [completionModels, setCompletionModels] = useState<ProviderModel[]>();
  const [embeddingModels, setEmbeddingModels] = useState<ProviderModel[]>();

  const common_tips = useTranslations('common.tips');
  const common_action = useTranslations('common.action');
  const page_collections = useTranslations('page_collections');
  const locale = useLocale();

  const defaultValues: FormValueType = {
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
      graph_backend_type: null,
      fulltext_backend_type: null,
      completion: {
        model_id: '',
        temperature: 0.1,
        tags: [],
      },
      embedding: {
        model_id: '',
        temperature: 0.1,
        tags: [],
      },
      language: locale as TitleLanguage,
    },
  };

  const CollectionConfigIndexTypes = {
    'config.enable_vector': {
      disabled: true,
      title: page_collections('index_type_VECTOR.title'),
      description: page_collections('index_type_VECTOR.description'),
    },
    'config.enable_fulltext': {
      disabled: true,
      title: page_collections('index_type_FULLTEXT.title'),
      description: page_collections('index_type_FULLTEXT.description'),
    },
    'config.enable_knowledge_graph': {
      disabled: false,
      title: page_collections('index_type_GRAPH.title'),
      description: page_collections('index_type_GRAPH.description'),
    },
    'config.enable_summary': {
      disabled: false,
      title: page_collections('index_type_SUMMARY.title'),
      description: page_collections('index_type_SUMMARY.description'),
    },
    'config.enable_vision': {
      disabled: false,
      title: page_collections('index_type_VISION.title'),
      description: page_collections('index_type_VISION.description'),
    },
  };

  const form = useForm<FormValueType>({
    resolver: zodResolver(collectionSchema),
    defaultValues:
      action === 'add' ? defaultValues : (collection as FormValueType),
  });

  /**
   * load models by 'enable_for_collection' in tags
   * set completion、embedding models used in model select component
   */
  const loadModels = useCallback(async () => {
    const models = await getAvailableModels(['chat', 'embedding']);
    setCompletionModels(
      [
        {
          label: page_collections('completion_model'),
          name: 'chat',
          models: models
            .filter((m) => m.capability === 'chat')
            .map((m) => ({
              model_id: m.id,
              display_name: m.display_name,
              temperature: 0.1,
              tags: [],
            })),
        },
      ],
    );
    setEmbeddingModels(
      [
        {
          label: page_collections('embedding_model'),
          name: 'embedding',
          models: models
            .filter((m) => m.capability === 'embedding')
            .map((m) => ({
              model_id: m.id,
              display_name: m.display_name,
              temperature: 0.1,
              tags: [],
            })),
        },
      ],
    );
  }, [page_collections]);

  /**
   * handle create or update a collection
   */
  const handleCreateOrUpdate = useCallback(
    async (values: FormValueType) => {
      if (action === 'edit') {
        if (!collection?.id) return;
        const data = await updateCollection(collection.id, values);
        if (data?.id) {
          toast.success(common_tips('update_success'));
          loadCollection();
        }
      }
      if (action === 'add') {
        const data = await createCollection(values);
        if (data?.id) {
          toast.success(common_tips('create_success'));
          router.push('/workspace/collections');
        }
      }
    },
    [action, collection.id, common_tips, loadCollection, router],
  );

  /**
   * Watch completionModelName
   * When the completion model name is changed, synchronize changes to other model parameters.
   */
  const completionModelName = useWatch({
    control: form.control,
    name: 'config.completion.model_id',
  });
  useEffect(() => {
    if (_.isEmpty(completionModels)) return;

    let defaultModel: ModelSpec | undefined;
    let currentModel: ModelSpec | undefined;
    let defaultProvider: ProviderModel | undefined;
    let currentProvider: ProviderModel | undefined;
    completionModels?.forEach((provider) => {
      provider.models?.forEach((m) => {
        if (m.model_id === completionModelName) {
          currentModel = m;
          currentProvider = provider;
        }
      });
    });

    form.setValue(
      'config.completion.model_id',
      currentModel?.model_id || defaultModel?.model_id || '',
    );
  }, [completionModelName, completionModels, form]);

  /**
   * Watch embeddingModelName
   * When the embedding model name is changed, synchronize changes to other model parameters.
   * In edit mode the embedding binding is immutable (backend rejects changes), so we
   * keep whatever values the collection already has and skip the auto-synchronisation
   * to avoid accidentally overwriting them with a default when the current model is
   * no longer present in the available models list.
   */
  const embeddingModelName = useWatch({
    control: form.control,
    name: 'config.embedding.model_id',
  });
  useEffect(() => {
    if (action === 'edit') return;
    if (_.isEmpty(embeddingModels)) return;

    let defaultModel: ModelSpec | undefined;
    let currentModel: ModelSpec | undefined;
    let defaultProvider: ProviderModel | undefined;
    let currentProvider: ProviderModel | undefined;

    embeddingModels?.forEach((provider) => {
      provider.models?.forEach((m) => {
        if (m.model_id === embeddingModelName) {
          currentModel = m;
          currentProvider = provider;
        }
      });
    });
    form.setValue(
      'config.embedding.model_id',
      currentModel?.model_id || defaultModel?.model_id || '',
    );
  }, [action, embeddingModelName, embeddingModels, form]);

  /**
   * load models
   */
  useEffect(() => {
    loadModels();
  }, [loadModels]);

  return (
    <>
      <Form {...form}>
        <form
          onSubmit={form.handleSubmit(handleCreateOrUpdate)}
          className="flex flex-col gap-5"
        >
          <Card className="gap-0 overflow-hidden rounded-xl border-border/70 py-0">
            <CardHeader className="border-border/70 bg-muted/60 border-b px-5 py-4">
              <div className="flex items-start gap-3">
                <div className="bg-accent-soft text-accent-ink flex size-10 shrink-0 items-center justify-center rounded-lg">
                  <Database className="size-5" />
                </div>
                <div>
                  <CardTitle className="text-base font-medium">
                    {page_collections('general')}
                  </CardTitle>
                  <CardDescription className="mt-1">
                    {page_collections('general_description')}
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="flex flex-col gap-6 px-5 py-5">
              <FormField
                control={form.control}
                name="title"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>{page_collections('name')}</FormLabel>
                    <FormControl>
                      <Input
                        className="md:w-7/12"
                        placeholder={page_collections('name_placeholder')}
                        {...field}
                        value={field.value || ''}
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
                    <FormLabel>{page_collections('description')}</FormLabel>
                    <FormControl>
                      <Textarea
                        className="h-32 md:w-8/12"
                        placeholder={page_collections(
                          'description_placeholder',
                        )}
                        {...field}
                        value={field.value || ''}
                      />
                    </FormControl>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="config.language"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>{page_collections('language')}</FormLabel>
                    <FormControl>
                      <RadioGroup
                        value={field.value}
                        onValueChange={field.onChange}
                        className="mt-2 flex flex-row flex-wrap items-center gap-4"
                      >
                        <Label>
                          <RadioGroupItem value="zh-CN" />
                          {page_collections('language_zh_CN')}
                        </Label>
                        <Label>
                          <RadioGroupItem value="en-US" />
                          {page_collections('language_en_US')}
                        </Label>
                      </RadioGroup>
                    </FormControl>
                  </FormItem>
                )}
              />
            </CardContent>
          </Card>

          <Card className="gap-0 overflow-hidden rounded-xl border-border/70 py-0">
            <CardHeader className="border-border/70 bg-muted/60 border-b px-5 py-4">
              <CardTitle className="text-base font-medium">
                {page_collections('index_types')}
              </CardTitle>
              <CardDescription>
                {page_collections('index_types_description')}
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3 px-5 py-5 md:grid-cols-2">
              {objectKeys(CollectionConfigIndexTypes)
                .filter(isVisibleCollectionConfigKey)
                .map((key) => {
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
                              'has-[[aria-checked=true]]:bg-accent-soft/60 has-[[aria-checked=true]]:border-primary/30 flex h-full items-center gap-3 rounded-xl border p-4 transition-colors',
                              item.disabled
                                ? 'cursor-not-allowed'
                                : 'hover:bg-muted cursor-pointer',
                            )}
                          >
                            <div className="grid gap-2">
                              <div className="flex items-center gap-2 leading-none font-medium">
                                {item.title}
                                {item.disabled && (
                                  <Badge>
                                    {page_collections('required')}
                                  </Badge>
                                )}
                              </div>
                              <p className="text-muted-foreground text-sm font-medium">
                                {item.description}
                              </p>
                            </div>
                            <FormControl className="ml-auto">
                              <Switch
                                checked={Boolean(field.value)}
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

          <Card className="gap-0 overflow-hidden rounded-xl border-border/70 py-0">
            <CardHeader className="border-border/70 bg-muted/60 border-b px-5 py-4">
              <div className="flex items-start gap-3">
                <div className="bg-card text-primary flex size-9 shrink-0 items-center justify-center rounded-lg border">
                  <Sparkles className="size-4" />
                </div>
                <div>
                  <CardTitle className="text-base font-medium">
                    {page_collections('model_settings')}
                  </CardTitle>
                  <CardDescription>
                    {page_collections('model_settings_description')}
                  </CardDescription>
                </div>
              </div>
            </CardHeader>

            <CardContent className="flex flex-col gap-6 px-5 py-5">
              <FormField
                control={form.control}
                name="config.embedding.model_id"
                render={({ field }) => {
                  const embeddingLocked = action === 'edit';
                  return (
                    <FormItem>
                      <FormLabel>
                        {page_collections('embedding_model')}
                        {embeddingLocked && (
                          <Badge variant="secondary" className="ml-2">
                            {page_collections('embedding_model_locked_badge')}
                          </Badge>
                        )}
                      </FormLabel>
                      <FormControl className="ml-auto">
                        <Select
                          {...field}
                          onValueChange={field.onChange}
                          value={field.value || ''}
                          disabled={embeddingLocked}
                        >
                          <SelectTrigger
                            className={cn(
                              'w-full md:w-7/12',
                              embeddingLocked
                                ? 'cursor-not-allowed opacity-70'
                                : 'cursor-pointer',
                            )}
                          >
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
                                          key={model.model_id}
                                          value={model.model_id || ''}
                                        >
                                          {modelSelectLabel(model)}
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
                        {embeddingLocked
                          ? page_collections('embedding_model_locked_description')
                          : page_collections('embedding_model_description')}
                      </FormDescription>
                    </FormItem>
                  );
                }}
              />

              <Separator />

              <FormField
                control={form.control}
                name="config.completion.model_id"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>
                      {page_collections('completion_model')}
                    </FormLabel>
                    <FormControl className="ml-auto">
                      <Select
                        {...field}
                        onValueChange={field.onChange}
                        value={field.value || ''}
                      >
                        <SelectTrigger className="w-full cursor-pointer md:w-7/12">
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
                                        key={model.model_id}
                                        value={model.model_id || ''}
                                      >
                                        {modelSelectLabel(model)}
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
                      {page_collections('completion_model_description')}
                    </FormDescription>
                  </FormItem>
                )}
              />
            </CardContent>
          </Card>

          <div className="flex flex-col-reverse gap-3 border-border/70 bg-background/80 py-2 sm:flex-row sm:justify-end">
            {action === 'add' && (
              <Button variant="outline" asChild>
                <Link href="/workspace/collections">
                  <ArrowLeft className="size-4" />
                  {common_action('cancel')}
                </Link>
              </Button>
            )}
            <Button type="submit" className="cursor-pointer px-6">
              {action === 'add'
                ? page_collections('create_collection')
                : page_collections('update_collection')}
            </Button>
          </div>
        </form>
      </Form>
    </>
  );
};
