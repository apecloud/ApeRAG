'use client';

import { QuestionSetDetail } from '@/api';
import { useCollectionContext } from '@/components/providers/collection-provider';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardAction,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import _ from 'lodash';
import {
  ArrowLeft,
  BookOpen,
  EllipsisVertical,
  FileUp,
  ListPlus,
  Plus,
  SquarePen,
  Trash,
} from 'lucide-react';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { QuestionSetActions } from '../question-set-actions';
import { QuestionSetDelete } from '../question-set-delete';
import { QuestionActions } from './question-actions';
import { QuestionDelete } from './question-delete';
import { QuestionGenerate } from './question-generate';

export const QuestionsList = ({
  questionSet,
}: {
  questionSet: QuestionSetDetail;
}) => {
  const { collection } = useCollectionContext();
  const page_question_set = useTranslations('page_question_set');
  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-row items-center justify-between gap-4 font-bold">
        <div className="flex flex-1 flex-row items-center gap-2 truncate text-lg">
          <Button size="icon" variant="secondary">
            <Link href={`/workspace/collections/${collection.id}/questions`}>
              <ArrowLeft />
            </Link>
          </Button>
          {questionSet.name}
        </div>
        <div className="flex flex-row items-center gap-2">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button>
                <Plus />
                <span className="hidden md:inline">
                  {page_question_set('add_question')}
                </span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-56" align="start">
              <DropdownMenuGroup>
                <QuestionActions action="add" questionSet={questionSet}>
                  <DropdownMenuItem>
                    <ListPlus />
                    {page_question_set('add_question_manual')}
                  </DropdownMenuItem>
                </QuestionActions>

                <QuestionGenerate questionSet={questionSet}>
                  <DropdownMenuItem>
                    <BookOpen />
                    {page_question_set('add_question_generator')}
                  </DropdownMenuItem>
                </QuestionGenerate>

                <DropdownMenuItem disabled>
                  <FileUp /> {page_question_set('add_question_import')}
                </DropdownMenuItem>
              </DropdownMenuGroup>
            </DropdownMenuContent>
          </DropdownMenu>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button size="icon" variant="outline">
                <EllipsisVertical />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48">
              <QuestionSetActions action="edit" questionSet={questionSet}>
                <DropdownMenuItem>
                  <SquarePen /> {page_question_set('update_question_set')}
                </DropdownMenuItem>
              </QuestionSetActions>
              <DropdownMenuSeparator />
              <QuestionSetDelete questionSet={questionSet}>
                <DropdownMenuItem variant="destructive">
                  <Trash /> {page_question_set('delete_question_set')}
                </DropdownMenuItem>
              </QuestionSetDelete>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {_.isEmpty(questionSet.questions) ? (
        <div className="bg-accent/50 text-muted-foreground rounded-lg py-40 text-center">
          {page_question_set('no_question_found')}
        </div>
      ) : (
        questionSet.questions?.map((question) => {
          return (
            <Card key={question.id}>
              <CardHeader>
                <CardTitle>{question.question_text}</CardTitle>
                <CardDescription>{question.ground_truth}</CardDescription>
                <CardAction>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button size="icon" variant="ghost">
                        <EllipsisVertical />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end" className="w-48">
                      <QuestionActions
                        action="edit"
                        question={question}
                        questionSet={questionSet}
                      >
                        <DropdownMenuItem>
                          <SquarePen /> {page_question_set('update_question')}
                        </DropdownMenuItem>
                      </QuestionActions>
                      <DropdownMenuSeparator />
                      <QuestionDelete
                        questionSet={questionSet}
                        question={question}
                      >
                        <DropdownMenuItem variant="destructive">
                          <Trash /> {page_question_set('delete_question')}
                        </DropdownMenuItem>
                      </QuestionDelete>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </CardAction>
              </CardHeader>
            </Card>
          );
        })
      )}
    </div>
  );
};
