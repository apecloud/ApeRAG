import { Button, Form, Input, Select, App } from 'antd';
import { useIntl, history, useModel } from 'umi';
import { useEffect, useState } from 'react';
import { EvaluationApi } from '@/api/apis/evaluation-api';
import { PageContainer } from '@/components/page-container';
import { PageHeader } from '@/components/page-header';
import { ModelSelect } from '@/components';

const { Option } = Select;
const evaluationApi = new EvaluationApi();

const NewEvaluationContent = () => {
  const intl = useIntl();
  const { message } = App.useApp();
  const [form] = Form.useForm();
  const [namePlaceholder, setNamePlaceholder] = useState(
    intl.formatMessage({ id: 'evaluation.new.form.name.placeholder.auto' }),
  );
  const { collections, getCollections } = useModel('collection');
  const { questionSets, refresh: refreshQuestionSets } = useModel('questionSet');
  const { getProviderByModelName, getAvailableModels } = useModel('models');

  useEffect(() => {
    getCollections();
    refreshQuestionSets();
    getAvailableModels();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const getFormattedTimestamp = () => {
    const now = new Date();
    const year = now.getFullYear();
    const month = (now.getMonth() + 1).toString().padStart(2, '0');
    const day = now.getDate().toString().padStart(2, '0');
    const hours = now.getHours().toString().padStart(2, '0');
    const minutes = now.getMinutes().toString().padStart(2, '0');
    const seconds = now.getSeconds().toString().padStart(2, '0');
    return `${year}${month}${day}${hours}${minutes}${seconds}`;
  };

  const handleValuesChange = (changedValues: any, allValues: any) => {
    const { collection_id, question_set_id } = allValues;

    if (collection_id && question_set_id) {
      const selectedCollection = collections?.find(
        (c) => c.id === collection_id,
      );
      const selectedQuestionSet = questionSets?.find(
        (qs) => qs.id === question_set_id,
      );

      if (selectedCollection && selectedQuestionSet) {
        const newName = `${
          selectedCollection.title
        }-${selectedQuestionSet.name} ${getFormattedTimestamp()}`;
        setNamePlaceholder(newName);
      }
    } else {
      setNamePlaceholder(
        intl.formatMessage({ id: 'evaluation.new.form.name.placeholder.auto' }),
      );
    }
  };

  const onFinish = async (values: any) => {
    const { agentLlm, judgeLlm, name, ...rest } = values;
    const { provider: agentProvider, model: agentModel } =
      getProviderByModelName(agentLlm, 'completion');
    const { provider: judgeProvider, model: judgeModel } =
      getProviderByModelName(judgeLlm, 'completion');

    if (!agentModel || !judgeModel || !agentProvider || !judgeProvider) {
      message.error('Agent LLM or Judge LLM not found');
      return;
    }

    const payload = {
      ...rest,
      name: name || namePlaceholder,
      agent_llm_config: {
        model_name: agentModel.model,
        model_service_provider: agentProvider.name,
        custom_llm_provider: agentModel.custom_llm_provider,
      },
      judge_llm_config: {
        model_name: judgeModel.model,
        model_service_provider: judgeProvider.name,
        custom_llm_provider: judgeModel.custom_llm_provider,
      },
    };

    try {
      const res = await evaluationApi.createEvaluationApiV1EvaluationsPost({
        evaluationCreate: payload,
      });
      message.success(
        intl.formatMessage({ id: 'tips.create.success' }),
      );
      const newEvaluation = res.data;
      if (newEvaluation.id) {
        history.push(`/evaluations/${newEvaluation.id}`);
      } else {
        history.push('/evaluations');
      }
    } catch (error) {
      message.error(intl.formatMessage({ id: 'tips.create.failed' }));
    }
  };

  return (
    <PageContainer>
      <PageHeader
        title={intl.formatMessage({ id: 'evaluation.new' })}
      />
      <Form
        form={form}
        layout="vertical"
        onFinish={onFinish}
        onValuesChange={handleValuesChange}
        style={{ maxWidth: 800 }}
      >
        <Form.Item
          name="name"
          label={intl.formatMessage({ id: 'evaluation.new.form.name' })}
        >
          <Input placeholder={namePlaceholder} />
        </Form.Item>
        <Form.Item
          name="collection_id"
            label={intl.formatMessage({
              id: 'evaluation.new.form.collection',
            })}
            rules={[{ required: true }]}
          >
            <Select
              placeholder={intl.formatMessage({
                id: 'evaluation.new.form.collection.placeholder',
              })}
            >
              {collections?.map((c) => (
                <Option key={c.id} value={c.id!}>
                  {c.title}
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="question_set_id"
            label={intl.formatMessage({
              id: 'evaluation.new.form.questionSet',
            })}
            rules={[{ required: true }]}
          >
            <Select
              placeholder={intl.formatMessage({
                id: 'evaluation.new.form.questionSet.placeholder',
              })}
              dropdownRender={(menu) =>
                questionSets && questionSets.length > 0 ? (
                  menu
                ) : (
                  <div style={{ padding: 8, textAlign: 'center' }}>
                    <p>
                      {intl.formatMessage({
                        id: 'evaluation.new.form.questionSet.empty',
                      })}
                    </p>
                    <Button
                      type="primary"
                      onClick={() =>
                        history.push('/evaluations/question-sets/new')
                      }
                    >
                      {intl.formatMessage({
                        id: 'evaluation.new.form.questionSet.create',
                      })}
                    </Button>
                  </div>
                )
              }
            >
              {questionSets?.map((qs) => (
                <Option key={qs.id} value={qs.id!}>
                  {qs.name}
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="agentLlm"
            label={intl.formatMessage({
              id: 'evaluation.new.form.agentLlm',
            })}
            rules={[{ required: true }]}
          >
            <ModelSelect
              model="completion"
              placeholder={intl.formatMessage({
                id: 'evaluation.new.form.agentLlm.placeholder',
              })}
              tagfilters={[
                {
                  operation: 'OR',
                  tags: ['enable_for_collection'],
                },
                {
                  operation: 'OR',
                  tags: ['enable_for_agent'],
                },
              ]}
            />
          </Form.Item>

          <Form.Item
            name="judgeLlm"
            label={intl.formatMessage({
              id: 'evaluation.new.form.judgeLlm',
            })}
            rules={[{ required: true }]}
          >
            <ModelSelect
              model="completion"
              placeholder={intl.formatMessage({
                id: 'evaluation.new.form.judgeLlm.placeholder',
              })}
              tagfilters={[
                {
                  operation: 'OR',
                  tags: ['enable_for_collection'],
                },
                {
                  operation: 'OR',
                  tags: ['enable_for_agent'],
                },
              ]}
            />
          </Form.Item>

          <Form.Item>
          <Button
            type="primary"
            htmlType="submit"
            style={{ marginRight: 8 }}
          >
            {intl.formatMessage({ id: 'evaluation.new.form.submit' })}
          </Button>
          <Button onClick={() => history.push('/evaluations')}>
            {intl.formatMessage({ id: 'evaluation.new.form.cancel' })}
          </Button>
        </Form.Item>
      </Form>
    </PageContainer>
  );
};

const NewEvaluation = () => (
  <App>
    <NewEvaluationContent />
  </App>
);

export default NewEvaluation;
