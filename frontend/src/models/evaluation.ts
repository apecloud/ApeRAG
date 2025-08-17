import { EvaluationApi } from '@/api';
import { Evaluation, EvaluationDetail } from '@/api/models';
import { useCallback, useState } from 'react';

const evaluationApi = new EvaluationApi();

export default () => {
  const [evaluations, setEvaluations] = useState<Evaluation[]>();
  const [currentEvaluation, setCurrentEvaluation] = useState<EvaluationDetail>();
  const [loading, setLoading] = useState(false);
  const [evaluationsLoading, setEvaluationsLoading] = useState(false);

  const getEvaluation = useCallback(async (id: string) => {
    setLoading(true);
    try {
      const { data } =
        await evaluationApi.getEvaluationApiV1EvaluationsEvalIdGet({
          evalId: id,
        });
      setCurrentEvaluation(data as EvaluationDetail);
      return data;
    } catch (e) {
      // handle error
    } finally {
      setLoading(false);
    }
  }, []);

  const getEvaluations = async () => {
    setEvaluationsLoading(true);
    try {
      const { data } = await evaluationApi.listEvaluationsApiV1EvaluationsGet();
      setEvaluations(data.items as Evaluation[]);
    } catch (e) {
      // handle error
    } finally {
      setEvaluationsLoading(false);
    }
  };

  return {
    evaluations,
    evaluationsLoading,
    getEvaluations,
    currentEvaluation,
    loading,
    getEvaluation,
  };
};
