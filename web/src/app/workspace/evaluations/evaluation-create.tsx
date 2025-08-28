'use client';

import { Collection, ModelSpec, QuestionSet } from '@/api';
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
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { apiClient } from '@/lib/api/client';
import { zodResolver } from '@hookform/resolvers/zod';
import { Slot } from '@radix-ui/react-slot';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useCallback, useEffect, useState } from 'react';
import { useForm, useWatch } from 'react-hook-form';
import { toast } from 'sonner';
import * as z from 'zod';

const evaluationSchema = z.object({
  name: z.string().min(1),
  collection_id: z.string().min(1),
  question_set_id: z.string().min(1),
  agent_llm_config: z.object({
    model_name: z.string().min(1),
    model_service_provider: z.string().min(1),
    custom_llm_provider: z.string().min(1),
  }),
  judge_llm_config: z.object({
    model_name: z.string().min(1),
    model_service_provider: z.string().min(1),
    custom_llm_provider: z.string().min(1),
  }),
});

type ProviderModel = {
  label?: string;
  name?: string;
  models?: ModelSpec[];
};
export const EvaluationCreate = ({
  children,
}: {
  children?: React.ReactNode;
}) => {
  const [visible, setVisible] = useState<boolean>(false);
  const [collections, setCollections] = useState<Collection[]>([]);
  const [questionSets, setQuestionSets] = useState<QuestionSet[]>([]);
  const [agentModels, setAgentModels] = useState<ProviderModel[]>([]);
  const router = useRouter();

  const form = useForm<z.infer<typeof evaluationSchema>>({
    resolver: zodResolver(evaluationSchema),
    defaultValues: {
      name: '',
      collection_id: '',
      question_set_id: '',
      agent_llm_config: {
        model_name: '',
        model_service_provider: '',
        custom_llm_provider: '',
      },
      judge_llm_config: {
        model_name: '',
        model_service_provider: '',
        custom_llm_provider: '',
      },
    },
  });
  const agentModelName = useWatch({
    control: form.control,
    name: 'agent_llm_config.model_name',
  });
  const judgeModelName = useWatch({
    control: form.control,
    name: 'judge_llm_config.model_name',
  });

  useEffect(() => {
    let currentAgentModel: ModelSpec | undefined;
    let currentAgentProvider: ProviderModel | undefined;
    let currentJudgeModel: ModelSpec | undefined;
    let currentJudgeProvider: ProviderModel | undefined;
    agentModels?.forEach((provider) => {
      provider.models?.forEach((m) => {
        if (m.model === agentModelName) {
          currentAgentModel = m;
          currentAgentProvider = provider;
        }
        if (m.model === judgeModelName) {
          currentJudgeModel = m;
          currentJudgeProvider = provider;
        }
      });
    });
    form.setValue(
      'agent_llm_config.custom_llm_provider',
      currentAgentModel?.custom_llm_provider || '',
    );
    form.setValue(
      'agent_llm_config.model_service_provider',
      currentAgentProvider?.name || '',
    );
    form.setValue(
      'judge_llm_config.custom_llm_provider',
      currentJudgeModel?.custom_llm_provider || '',
    );
    form.setValue(
      'judge_llm_config.model_service_provider',
      currentJudgeProvider?.name || '',
    );
  }, [agentModelName, judgeModelName, agentModels, form]);

  const loadData = useCallback(async () => {
    const [collectionRes, questionSetRes, agentModelsRes] = await Promise.all([
      apiClient.defaultApi.collectionsGet({
        page: 1,
        pageSize: 100,
        includeSubscribed: false,
      }),
      apiClient.evaluationApi.listQuestionSetsApiV1QuestionSetsGet({
        page: 1,
        pageSize: 100,
      }),
      apiClient.defaultApi.availableModelsPost({
        tagFilterRequest: {
          tag_filters: [{ operation: 'AND', tags: ['enable_for_agent'] }],
        },
      }),
    ]);
    setCollections(collectionRes.data.items || []);
    setQuestionSets(questionSetRes.data.items || []);
    setAgentModels(
      agentModelsRes.data.items?.map((m) => ({
        label: m.label,
        name: m.name,
        models: m.completion,
      })) || [],
    );
  }, []);

  const handleCreate = useCallback(
    async (values: z.infer<typeof evaluationSchema>) => {
      await apiClient.evaluationApi.createEvaluationApiV1EvaluationsPost({
        evaluationCreate: values,
      });
      setVisible(false);
      setTimeout(router.refresh, 300);
      toast.success('Saved successfully.');
    },
    [router.refresh],
  );

  useEffect(() => {
    if (visible) {
      loadData();
      form.reset();
    }
  }, [form, loadData, visible]);

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
            onSubmit={form.handleSubmit(handleCreate)}
            className="space-y-8"
          >
            <DialogHeader>
              <DialogTitle>Create Evaluation</DialogTitle>
              <DialogDescription></DialogDescription>
            </DialogHeader>
            <FormField
              control={form.control}
              name="name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Evaluation Name</FormLabel>
                  <FormControl>
                    <Input placeholder="Enter the evaluation name" {...field} />
                  </FormControl>
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="collection_id"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>
                    <span>Collection</span>

                    <Link
                      className="text-muted-foreground hover:text-primary ml-auto underline"
                      href="/workspace/collections/new"
                    >
                      Create Collection
                    </Link>
                  </FormLabel>
                  <FormControl>
                    <Select {...field} onValueChange={field.onChange}>
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="Select a model" />
                      </SelectTrigger>
                      <SelectContent>
                        {collections.map((item) => {
                          return (
                            <SelectItem key={item.id} value={item.id || ''}>
                              {item.title}
                            </SelectItem>
                          );
                        })}
                      </SelectContent>
                    </Select>
                  </FormControl>
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="question_set_id"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>
                    <span>Question Set</span>
                    <Link
                      className="text-muted-foreground hover:text-primary ml-auto underline"
                      href="/workspace/evaluations/questions"
                    >
                      Create question set
                    </Link>
                  </FormLabel>
                  <FormControl>
                    <Select {...field} onValueChange={field.onChange}>
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="Select a model" />
                      </SelectTrigger>
                      <SelectContent>
                        {questionSets.map((item) => {
                          return (
                            <SelectItem key={item.id} value={item.id || ''}>
                              {item.name}
                            </SelectItem>
                          );
                        })}
                      </SelectContent>
                    </Select>
                  </FormControl>
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="agent_llm_config.model_name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Agent LLM</FormLabel>
                  <FormControl>
                    <Select {...field} onValueChange={field.onChange}>
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="Select a model" />
                      </SelectTrigger>
                      <SelectContent>
                        {agentModels.map((item) => {
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
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="judge_llm_config.model_name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Judge LLM</FormLabel>
                  <FormControl>
                    <Select {...field} onValueChange={field.onChange}>
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="Select a model" />
                      </SelectTrigger>
                      <SelectContent>
                        {agentModels.map((item) => {
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
                </FormItem>
              )}
            />

            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setVisible(false)}
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
};
