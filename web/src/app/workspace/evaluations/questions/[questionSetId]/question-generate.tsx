'use client';
import { Collection, ModelSpec, QuestionSet } from '@/api';
import { ProviderModel } from '@/app/workspace/collections/collection-form';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
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
import { Textarea } from '@/components/ui/textarea';
import { apiClient } from '@/lib/api/client';
import { zodResolver } from '@hookform/resolvers/zod';
import { Slot } from '@radix-ui/react-slot';
import { Bot, CircleQuestionMark, LoaderCircle } from 'lucide-react';

import { useRouter } from 'next/navigation';
import { useCallback, useEffect, useState } from 'react';
import { useForm, useWatch } from 'react-hook-form';
import { toast } from 'sonner';
import * as z from 'zod';

const defaultPrompt = `You are an expert at asking questions. Please read the following document carefully and generate two types of questions and their corresponding standard answers based on the content.

**Question Types:**
1.  **Factual Questions**: Questions that can be answered directly from the text.
2.  **Inferential Questions**: Questions that require reasoning, comparison, or summarization of multiple pieces of information from the text to answer.

**Document Content:**

{DOCUMENT_CONTENT}

**Your Task:**
Please generate {NUMBER_OF_QUESTIONS} questions based on the document above. The number of factual and inferential questions should be equal. The language of the questions should be consistent with the language of the document. Please output a list of questions in JSON format. Each question object should contain three fields: \`question_type\` ('FACTUAL' or 'INFERENTIAL'), \`question_text\` (the content of the question), and \`ground_truth\` (the standard answer based on the document content).

**IMPORTANT**: Your response should only contain the JSON object, with no other text or explanations.

**Output Example:**
[
  {
    "question_type": "FACTUAL",
    "question_text": "What year was the project mentioned in the document launched?",
    "ground_truth": "According to the document, the project was officially launched in 2021."
  },
  {
    "question_type": "INFERENTIAL",
    "question_text": "What are the main differences in challenges between the early and late stages of the project?",
    "ground_truth": "The main challenges in the early stages were technology selection and team building, while in the later stages, they shifted to system performance optimization and market promotion."
  }
]
`;

const generateSchema = z.object({
  collection_id: z.string().min(1),
  llm_config: z.object({
    model_name: z.string().min(1),
    custom_llm_provider: z.string().min(1),
    model_service_provider: z.string().min(1),
  }),
  question_count: z.coerce.number<number>().max(20).min(1),
  prompt: z.string().min(1),
});

export const QuestionGenerate = ({
  questionSet,
  children,
}: {
  questionSet: QuestionSet;
  children: React.ReactNode;
}) => {
  const [visible, setVisible] = useState<boolean>(false);
  const router = useRouter();
  const [loading, setLoading] = useState<boolean>(false);
  const [collections, setCollections] = useState<Collection[]>([]);
  const [agentModels, setAgentModels] = useState<ProviderModel[]>([]);
  const form = useForm<z.infer<typeof generateSchema>>({
    resolver: zodResolver(generateSchema),
    defaultValues: {
      collection_id: '',
      llm_config: {
        model_name: '',
        custom_llm_provider: '',
        model_service_provider: '',
      },
      question_count: 10,
      prompt: defaultPrompt,
    },
  });
  const agentModelName = useWatch({
    control: form.control,
    name: 'llm_config.model_name',
  });

  const handleGenerate = useCallback(
    async (values: z.infer<typeof generateSchema>) => {
      if (!questionSet?.id) return;
      setLoading(true);
      try {
        const generateRes =
          await apiClient.evaluationApi.generateQuestionSetApiV1QuestionSetsGeneratePost(
            {
              questionSetGenerate: values,
            },
            {
              timeout: 1000 * 60,
            },
          );
        const questions = generateRes.data.questions || [];
        await apiClient.evaluationApi.addQuestionsApiV1QuestionSetsQsIdQuestionsPost(
          {
            qsId: questionSet.id,
            questionsAdd: {
              questions: questions.map((question) => ({
                question_text: question.question_text || '',
                ground_truth: question.ground_truth || '',
                // question_type: question.question_type as QuestionType,
              })),
            },
          },
        );
        setVisible(false);
        setLoading(false);
        router.refresh();
      } catch (err) {
        console.log(err);
        toast.error('generate error.');
        setLoading(false);
      }
    },
    [questionSet?.id, router],
  );

  const loadData = useCallback(async () => {
    const [collectionRes, agentModelsRes] = await Promise.all([
      apiClient.defaultApi.collectionsGet({
        page: 1,
        pageSize: 100,
        includeSubscribed: false,
      }),
      apiClient.defaultApi.availableModelsPost({
        tagFilterRequest: {
          tag_filters: [{ operation: 'AND', tags: ['enable_for_agent'] }],
        },
      }),
    ]);
    setCollections(collectionRes.data.items || []);
    setAgentModels(
      agentModelsRes.data.items?.map((m) => ({
        label: m.label,
        name: m.name,
        models: m.completion,
      })) || [],
    );
  }, []);

  useEffect(() => {
    let currentAgentModel: ModelSpec | undefined;
    let currentAgentProvider: ProviderModel | undefined;

    agentModels?.forEach((provider) => {
      provider.models?.forEach((m) => {
        if (m.model === agentModelName) {
          currentAgentModel = m;
          currentAgentProvider = provider;
        }
      });
    });
    form.setValue(
      'llm_config.custom_llm_provider',
      currentAgentModel?.custom_llm_provider || '',
    );
    form.setValue(
      'llm_config.model_service_provider',
      currentAgentProvider?.name || '',
    );
  }, [agentModelName, agentModels, form]);

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
            onSubmit={form.handleSubmit(handleGenerate)}
            className="space-y-6"
          >
            <DialogHeader>
              <DialogTitle>Generate from Collection</DialogTitle>
              <DialogDescription></DialogDescription>
            </DialogHeader>

            <FormField
              control={form.control}
              name="collection_id"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Collection</FormLabel>
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
              name="llm_config.model_name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>LLM</FormLabel>
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
              name="question_count"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Question Count</FormLabel>
                  <FormControl>
                    <Input type="number" min={1} max={20} {...field} />
                  </FormControl>
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="prompt"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Prompt Template</FormLabel>
                  <FormControl>
                    <Textarea
                      {...field}
                      className="max-h-40 resize-none"
                      placeholder="Enter your prompt..."
                    />
                  </FormControl>

                  <Alert>
                    <CircleQuestionMark />
                    <AlertTitle>
                      The following variables will be replaced by the system:
                    </AlertTitle>
                    <AlertDescription className="text-xs">
                      {`{DOCUMENT_CONTENT}: The content of the document from the collection.`}
                    </AlertDescription>
                    <AlertDescription className="text-xs">
                      {`{NUMBER_OF_QUESTIONS}: The number of questions to be generated.`}
                    </AlertDescription>
                  </Alert>
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
              <Button type="submit" disabled={loading}>
                {loading ? <LoaderCircle className="animate-spin" /> : <Bot />}
                Generate
              </Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
};
