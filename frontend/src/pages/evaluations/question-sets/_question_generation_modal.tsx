import { DefaultApi } from '@/api';
import { Collection, LlmProvider } from '@/api/models';
import { Form, Input, InputNumber, Modal, Select } from 'antd';
import { useEffect, useState } from 'react';
import { useIntl } from 'umi';

interface QuestionGenerationModalProps {
  open: boolean;
  onCancel: () => void;
  onOk: (values: any) => void;
}

export const QuestionGenerationModal = ({
  open,
  onCancel,
  onOk,
}: QuestionGenerationModalProps) => {
  const [form] = Form.useForm();
  const { formatMessage } = useIntl();
  const [collections, setCollections] = useState<Collection[]>([]);
  const [llmProviders, setLlmProviders] = useState<LlmProvider[]>([]);
  const api = new DefaultApi();

  useEffect(() => {
    if (open) {
      api.collectionsGet().then((res: any) => {
        setCollections(res.data.data);
      });
      api.llmConfigurationGet().then((res: any) => {
        setLlmProviders(res.data.providers);
      });
    }
  }, [open]);

  const handleOk = () => {
    form
      .validateFields()
      .then((values) => {
        onOk(values);
        form.resetFields();
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
    >
      <Form form={form} layout="vertical" name="form_in_modal">
        <Form.Item
          name="collection_id"
          label={formatMessage({ id: 'collection.name' })}
          rules={[{ required: true }]}
        >
          <Select>
            {collections.map((c) => (
              <Select.Option key={c.id} value={c.id}>
                {c.title}
              </Select.Option>
            ))}
          </Select>
        </Form.Item>
        <Form.Item
          name="llm_provider_name"
          label={formatMessage({ id: 'model.provider.name' })}
          rules={[{ required: true }]}
        >
          <Select>
            {llmProviders.map((p) => (
              <Select.Option key={p.name} value={p.name}>
                {p.name}
              </Select.Option>
            ))}
          </Select>
        </Form.Item>
        <Form.Item
          name="question_count"
          label={formatMessage({ id: 'evaluation.question_sets.question_count' })}
          initialValue={5}
          rules={[{ required: true }]}
        >
          <InputNumber min={1} max={20} />
        </Form.Item>
        <Form.Item
          name="prompt"
          label={formatMessage({ id: 'model.prompt.template' })}
        >
          <Input.TextArea rows={5} />
        </Form.Item>
      </Form>
    </Modal>
  );
};
