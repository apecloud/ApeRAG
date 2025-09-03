'use client';

import { QuestionSet } from '@/api';
import { FormatDate } from '@/components/format-date';
import { useCollectionContext } from '@/components/providers/collection-provider';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Plus } from 'lucide-react';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { useState } from 'react';
import { QuestionSetActions } from './question-set-actions';

export const QuestionSetList = ({
  questionSets,
}: {
  questionSets: QuestionSet[];
}) => {
  const { collection } = useCollectionContext();
  const [searchValue, setSearchValue] = useState<string>('');
  const page_question_set = useTranslations('page_question_set');
  return (
    <div>
      <div className="mb-4 flex flex-row items-center">
        <div>
          <Input
            placeholder={page_question_set('search')}
            value={searchValue}
            onChange={(e) => setSearchValue(e.currentTarget.value)}
          />
        </div>
        <div className="ml-auto flex items-center gap-2">
          <QuestionSetActions action="add">
            <Button>
              <Plus />
              <span className="hidden md:inline">
                {page_question_set('add_question_set')}
              </span>
            </Button>
          </QuestionSetActions>
        </div>
      </div>

      {questionSets.length === 0 ? (
        <div className="bg-accent/50 text-muted-foreground rounded-lg py-40 text-center">
          No question set found
        </div>
      ) : (
        <div className="sm:grid-col-1 grid gap-6 md:grid-cols-1 lg:grid-cols-3">
          {questionSets
            .filter((questionSet) => {
              if (searchValue === '') return true;
              return questionSet.name?.match(new RegExp(searchValue));
            })
            .map((questionSet) => {
              return (
                <Link
                  key={questionSet.id}
                  href={`/workspace/collections/${collection.id}/questions/${questionSet.id}`}
                >
                  <Card className="hover:bg-accent/30 cursor-pointer gap-4 rounded-md py-4">
                    <CardHeader className="px-4">
                      <CardTitle className="h-5 truncate">
                        {questionSet.name || '--'}
                      </CardTitle>
                      <CardDescription className="truncate">
                        {questionSet.description || 'No description available'}
                      </CardDescription>
                    </CardHeader>
                    <CardFooter className="text-muted-foreground px-4 text-xs">
                      <div>
                        {questionSet.gmt_created && (
                          <FormatDate
                            datetime={new Date(questionSet.gmt_created)}
                          />
                        )}
                      </div>
                    </CardFooter>
                  </Card>
                </Link>
              );
            })}
        </div>
      )}
    </div>
  );
};
