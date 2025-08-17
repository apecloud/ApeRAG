import { EvaluationApi } from '@/api';
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

  return { questionSets, loading, refresh };
};
