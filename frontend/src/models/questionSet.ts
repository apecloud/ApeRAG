import { EvaluationApi } from '@/api/apis/evaluation-api';
import { QuestionSet } from '@/api/models';
import { useRequest } from 'ahooks';

export default () => {
  const evaluationApi = new EvaluationApi();
  const {
    data: questionSets,
    loading,
    refresh,
  } = useRequest(() =>
    evaluationApi
      .listQuestionSetsApiV1QuestionSetsGet({
        page: 1,
        pageSize: 100, // Fetch up to 100 question sets
      })
      .then((res) => res.data.items as QuestionSet[]),
  );

  const getQuestionSet = (id: string) => {
    return questionSets?.find((set) => set.id === id);
  };

  return { questionSets, loading, refresh, getQuestionSet };
};
