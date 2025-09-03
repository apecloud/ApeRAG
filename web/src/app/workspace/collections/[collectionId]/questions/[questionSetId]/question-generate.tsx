'use client';
import { ModelSpec, QuestionSet } from '@/api';
import { ProviderModel } from '@/app/workspace/collections/collection-form';
import { useCollectionContext } from '@/components/providers/collection-provider';
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
import { Bot, LoaderCircle } from 'lucide-react';
import { useLocale, useTranslations } from 'next-intl';

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

const defaultPromptCN = `你是一个善于提出问题的专家。请仔细阅读以下文档内容，并根据内容生成两种类型的问题和对应的标准答案。

**问题类型:**
1.  **事实性问题 (Factual)**: 可以直接从文本中找到明确答案的问题。
2.  **推理性问题 (Inferential)**: 需要结合文本中多个信息点进行推理、比较或总结才能得出答案的问题。

**文档内容:**

{DOCUMENT_CONTENT}

**你的任务:**
请根据以上文档，生成 {NUMBER_OF_QUESTIONS} 个问题。事实性问题和推理性问题的数量应各占一半。提问的语言应与文档的语言尽量保持一致。请以 JSON 格式输出一个问题列表，每个问题对象包含三个字段: \`question_type\` ('FACTUAL' 或 'INFERENTIAL'), \`question_text\` (问题内容), 和 \`ground_truth\` (基于文档内容得到的标准答案)。

**重要提示**: 你的回答应该只包含 JSON 对象，不应有任何其他文本或解释。

**输出示例:**
[
  {
    "question_type": "FACTUAL",
    "question_text": "文档中提到的项目启动于哪一年？",
    "ground_truth": "根据文档，该项目于2021年正式启动。"
  },
  {
    "question_type": "INFERENTIAL",
    "question_text": "项目早期阶段和后期阶段的主要挑战有何不同？",
    "ground_truth": "项目早期挑战主要是技术选型和团队组建，后期挑战则转变为系统性能优化和市场推广。"
  }
]
`;

const generateSchema = z.object({
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
  const page_question_set = useTranslations('page_question_set');
  const common_action = useTranslations('common.action');
  const [loading, setLoading] = useState<boolean>(false);
  const [agentModels, setAgentModels] = useState<ProviderModel[]>([]);
  const locale = useLocale();
  const { collection } = useCollectionContext();
  const form = useForm<z.infer<typeof generateSchema>>({
    resolver: zodResolver(generateSchema),
    defaultValues: {
      llm_config: {
        model_name: '',
        custom_llm_provider: '',
        model_service_provider: '',
      },
      question_count: 10,
      prompt: locale === 'zh-CN' ? defaultPromptCN : defaultPrompt,
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
              questionSetGenerate: {
                ...values,
                collection_id: collection.id || '',
              },
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
    [collection.id, questionSet.id, router],
  );

  const loadData = useCallback(async () => {
    const [agentModelsRes] = await Promise.all([
      apiClient.defaultApi.availableModelsPost({
        tagFilterRequest: {
          tag_filters: [{ operation: 'AND', tags: ['enable_for_agent'] }],
        },
      }),
    ]);
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
              <DialogTitle>
                {page_question_set('add_question_generator')}
              </DialogTitle>
              <DialogDescription></DialogDescription>
            </DialogHeader>

            <FormField
              control={form.control}
              name="llm_config.model_name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>{page_question_set('question_llm')}</FormLabel>
                  <FormControl>
                    <Select {...field} onValueChange={field.onChange}>
                      <SelectTrigger className="w-full">
                        <SelectValue
                          placeholder={page_question_set(
                            'question_llm_placeholder',
                          )}
                        />
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
                  <FormLabel>{page_question_set('question_count')}</FormLabel>
                  <FormControl>
                    <Input
                      type="number"
                      min={1}
                      max={20}
                      {...field}
                      placeholder={page_question_set(
                        'question_count_placeholder',
                      )}
                    />
                  </FormControl>
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="prompt"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>{page_question_set('prompt_template')}</FormLabel>
                  <FormControl>
                    <Textarea
                      {...field}
                      className="max-h-40 resize-none"
                      placeholder={page_question_set(
                        'prompt_template_placeholder',
                      )}
                    />
                  </FormControl>

                  <div className="mt-4 text-sm">
                    {page_question_set('prompt_template_tips')}
                  </div>
                  <ul className="text-muted-foreground text-sm">
                    <li className="ml-4 list-disc">{`{DOCUMENT_CONTENT}: ${page_question_set('prompt_template_document_content')}`}</li>
                    <li className="ml-4 list-disc">{`{NUMBER_OF_QUESTIONS}: ${page_question_set('prompt_template_number_of_question')}`}</li>
                  </ul>
                </FormItem>
              )}
            />

            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setVisible(false)}
              >
                {common_action('cancel')}
              </Button>
              <Button type="submit" disabled={loading}>
                {loading ? <LoaderCircle className="animate-spin" /> : <Bot />}
                {common_action('continue')}
              </Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
};
