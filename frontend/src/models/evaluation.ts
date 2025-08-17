import { EvaluationApi } from '@/api/apis/evaluation-api';
import { Evaluation, EvaluationDetail } from '@/api/models';
import { useCallback, useState } from 'react';

const evaluationApi = new EvaluationApi();

export default () => {
  const [evaluations, setEvaluations] = useState<Evaluation[]>();
  const [currentEvaluation, setCurrentEvaluation] = useState<EvaluationDetail>();
  const [loading, setLoading] = useState(false);
  const [evaluationsLoading, setEvaluationsLoading] = useState(false);

  const getEvaluation = useCallback(
    async (id: string, options?: { background: boolean }) => {
      if (!options?.background) {
        setLoading(true);
      }
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
        if (!options?.background) {
          setLoading(false);
        }
      }
    },
    [],
  );

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

  const deleteEvaluation = useCallback(async (id: string) => {
    setLoading(true);
    try {
      await evaluationApi.deleteEvaluationApiV1EvaluationsEvalIdDelete({
        evalId: id,
      });
      return true;
    } catch (e) {
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  const pauseEvaluation = useCallback(
    async (id: string) => {
      setLoading(true);
      try {
        await evaluationApi.pauseEvaluationApiV1EvaluationsEvalIdPausePost({
          evalId: id,
        });
        await getEvaluation(id);
        return true;
      } catch (e) {
        return false;
      } finally {
        setLoading(false);
      }
    },
    [getEvaluation],
  );

  const resumeEvaluation = useCallback(
    async (id: string) => {
      setLoading(true);
      try {
        await evaluationApi.resumeEvaluationApiV1EvaluationsEvalIdResumePost({
          evalId: id,
        });
        await getEvaluation(id);
        return true;
      } catch (e) {
        return false;
      } finally {
        setLoading(false);
      }
    },
    [getEvaluation],
  );

  const retryEvaluation = useCallback(
    async (id: string, options: { scope: 'failed' | 'all' }) => {
      setLoading(true);
      try {
        await evaluationApi.retryEvaluationApiV1EvaluationsEvalIdRetryPost({
          evalId: id,
          scope: options.scope,
        });
        await getEvaluation(id);
        return true;
      } catch (e) {
        return false;
      } finally {
        setLoading(false);
      }
    },
    [getEvaluation],
  );

  return {
    evaluations,
    evaluationsLoading,
    getEvaluations,
    currentEvaluation,
    loading,
    getEvaluation,
    deleteEvaluation,
    pauseEvaluation,
    resumeEvaluation,
    retryEvaluation,
  };
};
