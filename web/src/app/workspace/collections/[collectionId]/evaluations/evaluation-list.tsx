'use client';

import { Evaluation } from '@/api';
import { FormatDate } from '@/components/format-date';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Separator } from '@/components/ui/separator';
import { cn } from '@/lib/utils';
import _ from 'lodash';
import { Plus } from 'lucide-react';
import Link from 'next/link';

import { useCollectionContext } from '@/components/providers/collection-provider';
import { useTranslations } from 'next-intl';
import { useState } from 'react';
import { EvaluationCreate } from './evaluation-create';

export const EvaluationList = ({
  evaluations,
}: {
  evaluations: Evaluation[];
}) => {
  const [searchValue, setSearchValue] = useState<string>('');
  const page_evaluation = useTranslations('page_evaluation');
  const { collection } = useCollectionContext();
  return (
    <>
      <div className="mb-4 flex flex-row items-center">
        <div>
          <Input
            placeholder={page_evaluation('search')}
            value={searchValue}
            onChange={(e) => setSearchValue(e.currentTarget.value)}
          />
        </div>
        <div className="ml-auto flex items-center gap-2">
          <EvaluationCreate>
            <Button>
              <Plus />
              <span className="hidden md:inline">
                {page_evaluation('add_evaluation')}
              </span>
            </Button>
          </EvaluationCreate>
        </div>
      </div>

      {evaluations.length === 0 ? (
        <div className="bg-accent/50 text-muted-foreground rounded-lg py-40 text-center">
          No evaluation found
        </div>
      ) : (
        <div className="sm:grid-col-1 grid gap-6 md:grid-cols-1 lg:grid-cols-3">
          {evaluations
            .filter((evaluation) => {
              if (searchValue === '') return true;
              return evaluation.name?.match(new RegExp(searchValue));
            })
            .map((evaluation) => {
              return (
                <Link
                  key={evaluation.id}
                  href={`/workspace/collections/${collection.id}/evaluations/${evaluation.id}`}
                >
                  <Card className="hover:bg-accent/30 cursor-pointer gap-4 rounded-md py-4">
                    <CardHeader className="px-4">
                      <CardTitle className="h-5 truncate">
                        {evaluation.name || '--'}
                      </CardTitle>
                    </CardHeader>

                    <CardContent className="flex flex-row justify-between">
                      <div className="w-6/12 text-center">
                        <div className="text-muted-foreground text-sm">
                          {page_evaluation('questions')}
                        </div>
                        <div
                          className={cn(
                            'flex flex-row justify-center gap-1 font-bold',

                            evaluation.total_questions &&
                              evaluation.total_questions > 100
                              ? 'text-lg'
                              : 'text-xl',
                          )}
                        >
                          <span
                            className={
                              evaluation.completed_questions !==
                              evaluation.total_questions
                                ? 'text-muted-foreground'
                                : undefined
                            }
                          >
                            {evaluation.completed_questions ?? '-'}
                          </span>
                          /<span>{evaluation.total_questions ?? '-'}</span>
                        </div>
                      </div>
                      <Separator
                        orientation="vertical"
                        className="data-[orientation=vertical]:h-10"
                      />
                      <div className="w-6/12 text-center">
                        <div className="text-muted-foreground text-sm">
                          {page_evaluation('avg_score')}
                        </div>
                        <div className="text-xl font-bold">
                          {evaluation.average_score?.toFixed(2) ?? '-'}
                        </div>
                      </div>
                    </CardContent>
                    <Separator />
                    <CardFooter className="text-muted-foreground px-4 text-xs">
                      <div>
                        {evaluation.gmt_created && (
                          <FormatDate
                            datetime={new Date(evaluation.gmt_created)}
                          />
                        )}
                      </div>

                      <div className="ml-auto flex flex-row items-center gap-2">
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
                    </CardFooter>
                  </Card>
                </Link>
              );
            })}
        </div>
      )}
    </>
  );
};
