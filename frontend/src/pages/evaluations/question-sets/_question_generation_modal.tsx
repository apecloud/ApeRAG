import { EvaluationApi } from '@/api/apis/evaluation-api';
import { DefaultApi } from '@/api/apis/default-api';
import { Collection } from '@/api/models';
import { ModelSelect } from '@/components';
import { InfoCircleOutlined } from '@ant-design/icons';
import { App, Form, Input, InputNumber, Modal, Popover, Select, Space } from 'antd';
import { useEffect, useState } from 'react';
import { useIntl, useModel } from 'umi';

interface QuestionGenerationModalProps {
  open: boolean;
  onCancel: () => void;
  onGenerated: (questions: any[]) => void;
}

export const QuestionGenerationModal = ({
  open,
  onCancel,
  onGenerated,
}: QuestionGenerationModalProps) => {
  const [form] = Form.useForm();
  const { formatMessage } = useIntl();
  const { message } = App.useApp();
  const { getProviderByModelName, getAvailableModels } = useModel('models');
  const [collections, setCollections] = useState<Collection[]>([]);
  const [loading, setLoading] = useState(false);
  const evaluationApi = new EvaluationApi();
  const defaultApi = new DefaultApi();

  useEffect(() => {
    if (open) {
      getAvailableModels();
      setLoading(true);
      defaultApi
        .collectionsGet()
        .then((collectionsRes) => {
          setCollections((collectionsRes.data as any).items || []);
        })
        .catch(() => {
          message.error(formatMessage({ id: 'tips.get.failed' }));
        })
        .finally(() => {
          setLoading(false);
        });
    }
  }, [open]);

  const handleOk = () => {
    form
      .validateFields()
      .then((values) => {
        setLoading(true);
        const { llm_model_name, ...rest } = values;
        const { provider, model } = getProviderByModelName(
          llm_model_name,
          'completion',
        );
        const llm_config = {
          model_name: model?.model,
          model_service_provider: provider?.name,
          custom_llm_provider: model?.custom_llm_provider,
        };

        evaluationApi
          .generateQuestionSetApiV1QuestionSetsGeneratePost({
            questionSetGenerate: { ...rest, llm_config },
          })
          .then((res: any) => {
            onGenerated(res.data.questions || []);
            form.resetFields();
          })
          .catch((e) => {
            const errorMsg =
              e?.response?.data?.detail || e?.response?.data?.message || e.message;
            message.error(
              `${formatMessage({ id: 'tips.generate.failed' })}: ${errorMsg}`,
            );
          })
          .finally(() => {
            setLoading(false);
          });
      })
      .catch((info) => {
        console.log('Validate Failed:', info);
      });
  };

  return (
    <Modal
      title={formatMessage({
        id: 'evaluation.question_sets.generate_from_collection',
      })}
      open={open}
      onOk={handleOk}
      onCancel={onCancel}
      destroyOnClose
      confirmLoading={loading}
    >
      <Form form={form} layout="vertical" name="form_in_modal">
        <Form.Item
          name="collection_id"
          label={formatMessage({ id: 'collection.name' })}
          rules={[{ required: true }]}
        >
          <Select loading={loading}>
            {collections.map((c) => (
              <Select.Option key={c.id} value={c.id}>
                {c.title}
              </Select.Option>
            ))}
          </Select>
        </Form.Item>
        <Form.Item
          name="llm_model_name"
          label={formatMessage({ id: 'model.name' })}
          rules={[{ required: true }]}
        >
          <ModelSelect
            model="completion"
            tagfilters={[
              {
                operation: 'OR',
                tags: ['enable_for_collection'],
              },
            ]}
          />
        </Form.Item>
        <Form.Item
          name="question_count"
          label={formatMessage({ id: 'evaluation.question_sets.question_count' })}
          initialValue={10}
          rules={[{ required: true }]}
        >
          <InputNumber min={1} max={20} />
        </Form.Item>
        <Form.Item
          label={
            <Space>
              {formatMessage({ id: 'model.prompt.template' })}
              <Popover
                content={
                  <div style={{ width: 300, whiteSpace: 'pre-wrap' }}>
                    {formatMessage({
                      id: 'evaluation.question_sets.prompt.template.hint',
                    })}
                  </div>
                }
              >
                <InfoCircleOutlined style={{ cursor: 'pointer' }} />
              </Popover>
            </Space>
          }
          name="prompt"
          initialValue={formatMessage({
            id: 'evaluation.question_sets.prompt.template.default',
          })}
        >
          <Input.TextArea rows={5} />
        </Form.Item>
      </Form>
    </Modal>
  );
};
