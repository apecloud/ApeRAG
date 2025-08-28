'use client';

import { QuestionSetDetail } from '@/api';
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
  BookOpen,
  EllipsisVertical,
  FileUp,
  ListPlus,
  Plus,
  SquarePen,
  Trash,
} from 'lucide-react';
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
  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-row items-center justify-between gap-4 font-bold">
        <div className="flex-1 truncate text-lg">{questionSet.name}</div>
        <div className="flex flex-row items-center gap-2">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button>
                <Plus />
                <span className="hidden md:inline">Add Questions</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-56" align="start">
              <DropdownMenuGroup>
                <QuestionActions action="add" questionSet={questionSet}>
                  <DropdownMenuItem>
                    <ListPlus />
                    Manual
                  </DropdownMenuItem>
                </QuestionActions>

                <QuestionGenerate questionSet={questionSet}>
                  <DropdownMenuItem>
                    <BookOpen />
                    Generate from Collection
                  </DropdownMenuItem>
                </QuestionGenerate>

                <DropdownMenuItem disabled>
                  <FileUp /> Import from File
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
                  <SquarePen /> Edit Question Set
                </DropdownMenuItem>
              </QuestionSetActions>
              <DropdownMenuSeparator />
              <QuestionSetDelete questionSet={questionSet}>
                <DropdownMenuItem variant="destructive">
                  <Trash /> Delete
                </DropdownMenuItem>
              </QuestionSetDelete>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {_.isEmpty(questionSet.questions) ? (
        <div className="bg-accent/50 text-muted-foreground rounded-lg py-40 text-center">
          No question found
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
                          <SquarePen /> Edit
                        </DropdownMenuItem>
                      </QuestionActions>
                      <DropdownMenuSeparator />
                      <QuestionDelete
                        questionSet={questionSet}
                        question={question}
                      >
                        <DropdownMenuItem variant="destructive">
                          <Trash /> Delete
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
