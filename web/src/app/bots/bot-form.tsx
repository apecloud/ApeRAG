'use client';

import { Bot, Collection, ModelSpec } from '@/api';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Combobox,
  ComboboxAnchor,
  ComboboxBadgeItem,
  ComboboxBadgeList,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxTrigger,
} from '@/components/ui/combobox';
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
import { Textarea } from '@/components/ui/textarea';
import { apiClient } from '@/lib/api/client';
import { zodResolver } from '@hookform/resolvers/zod';
import _ from 'lodash';
import { ChevronDown } from 'lucide-react';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useCallback, useEffect, useState } from 'react';
import { useForm, useWatch } from 'react-hook-form';
import { toast } from 'sonner';

import * as z from 'zod';

const botSchema = z.object({
  title: z.string().min(1),
  description: z.string(),
  config: z.object({
    agent: z.object({
      completion: z.object({
        model: z.string().min(1),
        model_service_provider: z.string().min(1),
        custom_llm_provider: z.string().min(1),
      }),
      system_prompt_template: z.string(),
      query_prompt_template: z.string(),
      collections: z.array(
        z.object({
          id: z.string(),
          title: z.string(),
        }),
      ),
    }),
  }),
});

type ProviderModel = {
  label?: string;
  name?: string;
  models?: ModelSpec[];
};

export const BotForm = ({
  bot,
  action,
}: {
  bot?: Bot;
  action: 'add' | 'edit';
}) => {
  const router = useRouter();
  const common_action = useTranslations('common.action');
  const common_tips = useTranslations('common.tips');
  const page_bot = useTranslations('page_bot');
  const [completionModels, setCompletionModels] = useState<ProviderModel[]>([]);
  const [collections, setCollections] = useState<Collection[]>([]);
  const form = useForm<z.infer<typeof botSchema>>({
    resolver: zodResolver(botSchema),
    defaultValues: {
      title: bot?.title || '',
      description: bot?.description || '',
      config: {
        agent: {
          completion: {
            model: bot?.config?.agent?.completion?.model || '',
            model_service_provider:
              bot?.config?.agent?.completion?.model_service_provider || '',
            custom_llm_provider:
              bot?.config?.agent?.completion?.custom_llm_provider || '',
          },
          system_prompt_template:
            bot?.config?.agent?.system_prompt_template || '',
          query_prompt_template:
            bot?.config?.agent?.query_prompt_template || '',
          collections: bot?.config?.agent?.collections || [],
        },
      },
    },
  });

  const completionModelName = useWatch({
    control: form.control,
    name: 'config.agent.completion.model',
  });

  useEffect(() => {
    if (completionModels.length === 0) return;

    let defaultModel: ModelSpec | undefined;
    let currentModel: ModelSpec | undefined;
    let defaultProvider: ProviderModel | undefined;
    let currentProvider: ProviderModel | undefined;

    completionModels?.forEach((provider) => {
      provider.models?.forEach((m) => {
        if (m.tags?.some((t) => t === 'default_for_agent_completion')) {
          defaultModel = m;
          defaultProvider = provider;
        }
        if (m.model === completionModelName) {
          currentModel = m;
          currentProvider = provider;
        }
      });
    });

    form.setValue(
      'config.agent.completion.custom_llm_provider',
      currentModel?.custom_llm_provider ||
        currentModel?.custom_llm_provider ||
        '',
    );

    form.setValue(
      'config.agent.completion.model_service_provider',
      currentProvider?.name || defaultProvider?.name || '',
    );
    form.setValue(
      'config.agent.completion.model',
      currentModel?.model || defaultModel?.model || '',
    );
  }, [completionModelName, completionModels, form]);

  const selectedCollections = useWatch({
    control: form.control,
    name: 'config.agent.collections',
  });

  const loadData = useCallback(async () => {
    const [modeRes, collectionsRes] = await Promise.all([
      apiClient.defaultApi.availableModelsPost({
        tagFilterRequest: {
          tag_filters: [{ operation: 'AND', tags: ['enable_for_agent'] }],
        },
      }),
      apiClient.defaultApi.collectionsGet({
        pageSize: 100,
        page: 1,
      }),
    ]);

    const completion = modeRes.data.items?.map((m) => {
      return {
        label: m.label,
        name: m.name,
        models: m.completion,
      };
    });

    setCollections(collectionsRes.data.items || []);
    setCompletionModels(completion || []);
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleCreateOrUpdate = useCallback(
    async (values: z.infer<typeof botSchema>) => {
      if (action === 'add') {
        const res = await apiClient.defaultApi.botsPost({
          botCreate: {
            ...values,
            type: 'agent',
          },
        });
        if (res.data.id) {
          router.push(`/bots/${res.data.id}/chats`);
        }
      }

      if (action === 'edit' && bot?.id) {
        const res = await apiClient.defaultApi.botsBotIdPut({
          botId: bot.id,
          botUpdate: values,
        });
        // @ts-expect-error openapi define error
        const id = res.data.id;
        if (id) {
          toast.success(common_tips('save_success'));
          router.refresh();
        }
      }
    },
    [action, bot?.id, common_tips, router],
  );

  return (
    <Form {...form}>
      <form
        onSubmit={form.handleSubmit(handleCreateOrUpdate)}
        className="my-4 space-y-8"
      >
        <Card>
          <CardHeader>
            <CardTitle>{page_bot('general')}</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-6">
            <FormField
              control={form.control}
              name="title"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>{page_bot('title')}</FormLabel>
                  <FormControl>
                    <Input
                      className="w-full md:w-6/12"
                      placeholder={page_bot('title_placeholder')}
                      {...field}
                    />
                  </FormControl>
                  <FormDescription></FormDescription>
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="description"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>{page_bot('description')}</FormLabel>
                  <FormControl>
                    <Textarea
                      className="h-30 w-full md:w-6/12"
                      placeholder={page_bot('description_placeholder')}
                      {...field}
                    />
                  </FormControl>
                  <FormDescription></FormDescription>
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="config.agent.collections"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>{page_bot('collection')}</FormLabel>
                  <FormControl>
                    <Combobox
                      value={field.value.map((item) => item.id)}
                      onValueChange={(v) => {
                        field.onChange(
                          collections.filter((col) =>
                            v.some((id) => id === col.id),
                          ),
                        );
                      }}
                      multiple
                      className="w-full cursor-pointer md:w-6/12"
                    >
                      <ComboboxTrigger className="w-full" asChild>
                        <ComboboxAnchor>
                          <ComboboxBadgeList>
                            {selectedCollections.map((item) => {
                              return (
                                <ComboboxBadgeItem
                                  key={item.id}
                                  value={item.id || ''}
                                >
                                  {item.title}
                                </ComboboxBadgeItem>
                              );
                            })}
                          </ComboboxBadgeList>
                          <ComboboxInput
                            placeholder={page_bot('collection_placeholder')}
                            className="h-auto min-w-20 flex-1"
                          />

                          <div className="absolute top-2 right-2">
                            <ChevronDown className="h-4 w-4 opacity-50" />
                          </div>
                        </ComboboxAnchor>
                      </ComboboxTrigger>
                      <ComboboxContent>
                        <ComboboxEmpty>
                          {page_bot('no_collection_found')}
                        </ComboboxEmpty>
                        {collections?.map((collection) => (
                          <ComboboxItem
                            key={collection.id}
                            value={collection.id || ''}
                          >
                            {collection.title}
                          </ComboboxItem>
                        ))}
                      </ComboboxContent>
                    </Combobox>
                  </FormControl>
                  <FormDescription></FormDescription>
                </FormItem>
              )}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>{page_bot('model_config')}</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-6">
            <FormField
              control={form.control}
              name="config.agent.completion.model"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>{page_bot('model')}</FormLabel>
                  <FormControl className="ml-auto">
                    <Select
                      {...field}
                      onValueChange={field.onChange}
                      value={field.value || ''}
                    >
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
                    {page_bot('model_description')}
                  </FormDescription>
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="config.agent.system_prompt_template"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>{page_bot('system_prompt_template')}</FormLabel>
                  <FormControl>
                    <Textarea
                      className="h-35"
                      placeholder={page_bot(
                        'system_prompt_template_placeholder',
                      )}
                      {...field}
                    />
                  </FormControl>
                  <FormDescription>
                    {page_bot('system_prompt_template_description')}
                  </FormDescription>
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="config.agent.query_prompt_template"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>{page_bot('query_prompt_template')}</FormLabel>
                  <FormControl>
                    <Textarea
                      className="h-35"
                      placeholder={page_bot(
                        'query_prompt_template_placeholder',
                      )}
                      {...field}
                    />
                  </FormControl>
                  <FormDescription>
                    {page_bot('query_prompt_template_description')}
                  </FormDescription>
                </FormItem>
              )}
            />
          </CardContent>
        </Card>

        <div className="flex">
          <div className="ml-auto flex flex-row gap-4">
            <Button variant="secondary" className="px-6" type="button" asChild>
              <Link href={bot ? `/bots/${bot.id}/chats` : '/bots'}>
                {common_action('cancel')}
              </Link>
            </Button>
            <Button className="cursor-pointer px-6" type="submit">
              {common_action('save')}
            </Button>
          </div>
        </div>
      </form>
    </Form>
  );
};
