import { FormatDate } from '@/components/format-date';
import { Markdown } from '@/components/markdown';
import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardAction,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Separator } from '@/components/ui/separator';
import { getServerApi } from '@/lib/api/server';
import { cn } from '@/lib/utils';
import _ from 'lodash';
import {
  ChevronRight,
  EllipsisVertical,
  ListRestart,
  RotateCcw,
  Trash,
} from 'lucide-react';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { EvaluationDeleteItem } from './evaluation-delete';
import { EvaluationRetryItem } from './evaluation-retry-item';

export default async function Page({
  params,
}: {
  params: Promise<{ evaluationId: string }>;
}) {
  const { evaluationId } = await params;
  const serverApi = await getServerApi();

  let evaluation;

  try {
    const [evaluationRes] = await Promise.all([
      serverApi.evaluationApi.getEvaluationApiV1EvaluationsEvalIdGet({
        evalId: evaluationId,
      }),
    ]);
    evaluation = evaluationRes.data;
  } catch (err) {
    console.log(err);
  }

  if (!evaluation) {
    notFound();
  }

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[
          { title: 'Evaluation', href: '/workspace/evaluations' },
          { title: evaluation.name || '--' },
        ]}
      />
      <PageContent>
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="text-2xl">
              {evaluation.name || '--'}
            </CardTitle>
            <CardDescription className="flex flex-row items-center gap-6 text-sm">
              {evaluation.gmt_created && (
                <FormatDate datetime={new Date(evaluation.gmt_created)} />
              )}

              <div className="flex flex-row items-center gap-2">
                <div
                  data-status={evaluation.status}
                  className={cn(
                    'size-2 rounded-lg',
                    'data-[status=COMPLETED]:bg-green-700',
                    'data-[status=FAILED]:bg-red-500',
                    'data-[status=PENDING]:bg-gray-500',
                    'data-[status=PAUSED]:bg-amber-500',
                    'data-[status=RUNNING]:bg-sky-500',
                  )}
                />
                {_.upperFirst(_.lowerCase(evaluation.status))}
              </div>

              <div className="flex flex-row gap-1">
                <span className="text-muted-foreground">Collection: </span>
                <Link
                  href={`/workspace/collections/${evaluation.config?.collection_id}/documents`}
                  className="hover:text-primary underline"
                >
                  {_.truncate(evaluation.collection_name, { length: 20 })}
                </Link>
              </div>

              <div className="flex flex-row gap-1">
                <span className="text-muted-foreground">Question Set: </span>
                <span>
                  <Link
                    href={`/workspace/evaluations/questions/${evaluation.config?.question_set_id}`}
                    className="hover:text-primary underline"
                  >
                    {_.truncate(evaluation.question_set_name, { length: 20 })}
                  </Link>
                </span>
              </div>
            </CardDescription>
            <CardAction className="flex flex-row gap-4">
              <div className="flex flex-row items-center">
                <span className="text-muted-foreground text-sm">
                  Avg. Score: &nbsp;
                </span>
                <span className="text-2xl font-bold">
                  {evaluation.average_score}
                </span>
              </div>

              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button size="icon" variant="ghost">
                    <EllipsisVertical />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-60">
                  <EvaluationRetryItem scope={'failed'} evaluation={evaluation}>
                    <ListRestart />
                    Retry Failed
                  </EvaluationRetryItem>
                  <EvaluationRetryItem scope="all" evaluation={evaluation}>
                    <RotateCcw />
                    Retry All
                  </EvaluationRetryItem>

                  <DropdownMenuSeparator />
                  <EvaluationDeleteItem evaluation={evaluation}>
                    <DropdownMenuItem variant="destructive">
                      <Trash /> Delete
                    </DropdownMenuItem>
                  </EvaluationDeleteItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </CardAction>
          </CardHeader>
        </Card>

        <div className="flex flex-col gap-4">
          {evaluation.items?.map((item, index) => {
            return (
              <Collapsible
                key={item.id}
                className="group/collapsible flex flex-col gap-2"
              >
                <CollapsibleTrigger asChild>
                  <Button
                    size="lg"
                    variant="secondary"
                    className="h-14 w-full cursor-pointer justify-start"
                  >
                    <ChevronRight className="transition-transform duration-200 group-data-[state=open]/collapsible:rotate-90" />
                    <span className="flex-1 truncate text-left">
                      {index + 1}. {item.question_text}
                    </span>

                    {item.status === 'COMPLETED' ? (
                      <div
                        data-score={item.llm_judge_score}
                        className={cn(
                          'ml-auto flex size-8 flex-col justify-center rounded-full bg-gray-500 text-center text-white',
                          'data-[score=5]:bg-green-700',
                          'data-[score=4]:bg-cyan-700',
                          'data-[score=3]:bg-amber-700',
                          'data-[score=2]:bg-fuchsia-700',
                          'data-[score=1]:bg-rose-700',
                        )}
                      >
                        {item.llm_judge_score}
                      </div>
                    ) : (
                      <div className="text-muted-foreground ml-auto flex flex-row items-center gap-2">
                        <div
                          data-status={item.status}
                          className={cn(
                            'size-2 rounded-lg bg-gray-500',
                            'data-[status=COMPLETED]:bg-green-700',
                            'data-[status=FAILED]:bg-red-500',
                            'data-[status=PENDING]:bg-gray-500',
                            'data-[status=RUNNING]:bg-sky-500',
                          )}
                        />
                        {_.upperFirst(_.lowerCase(item.status))}
                      </div>
                    )}
                  </Button>
                </CollapsibleTrigger>

                <CollapsibleContent className="flex flex-col gap-6 rounded-lg border p-6 text-sm">
                  <div>
                    <div className="text-muted-foreground mb-4">
                      Ground Truth
                    </div>
                    <div>{item.ground_truth}</div>
                  </div>
                  <Separator />
                  <div>
                    <div className="text-muted-foreground">RAG Answer</div>
                    <Markdown>{item.rag_answer}</Markdown>
                  </div>

                  <Separator />

                  <div>
                    <div className="text-muted-foreground">
                      LLM Judge Reasoning
                    </div>
                    <Markdown>{item.llm_judge_reasoning}</Markdown>
                  </div>
                </CollapsibleContent>
              </Collapsible>
            );
          })}
        </div>
      </PageContent>
    </PageContainer>
  );
}
