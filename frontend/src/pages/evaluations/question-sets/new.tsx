import { PageContainer } from '@/components/page-container';
import { PageHeader } from '@/components/page-header';
import {
  DeleteOutlined,
  InfoCircleOutlined,
  PlusOutlined,
  UploadOutlined,
} from '@ant-design/icons';
import { EvaluationApi } from '@/api/apis/evaluation-api';
import {
  App,
  Button,
  Divider,
  Flex,
  Form,
  Input,
  Popover,
  Space,
  Table,
  Tooltip,
  Upload,
} from 'antd';
import type { UploadProps } from 'antd';
import Papa from 'papaparse';
import { useEffect, useRef, useState } from 'react';
import { history, useIntl, useModel } from 'umi';
import { QuestionGenerationModal } from './_question_generation_modal';

const { TextArea } = Input;

interface QuestionItem {
  key: number;
  question: string;
  ground_truth: string;
}

const NewQuestionSetPage = () => {
  const { formatMessage } = useIntl();
  const { message } = App.useApp();
  const { refresh: refreshQuestionSets } = useModel('questionSet');
  const [form] = Form.useForm();
  const [questions, setQuestions] = useState<QuestionItem[]>([]);
  const [nextKey, setNextKey] = useState(1);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [editingKey, setEditingKey] = useState<number | null>(null);
  const [editingField, setEditingField] = useState<string | null>(null);
  const inputRef = useRef<any>(null);
  const evaluationApi = new EvaluationApi();

  useEffect(() => {
    if (editingKey !== null && inputRef.current) {
      inputRef.current.focus();
    }
  }, [editingKey, editingField]);

  const handleAddQuestion = () => {
    const newKey = nextKey;
    const newQuestion: QuestionItem = {
      key: newKey,
      question: '',
      ground_truth: '',
    };
    setQuestions([...questions, newQuestion]);
    setNextKey(newKey + 1);
    setEditingKey(newKey);
    setEditingField('question');
  };

  const handleDeleteQuestion = (key: number) => {
    setQuestions(questions.filter((item) => item.key !== key));
  };

  const handleQuestionChange = (
    key: number,
    field: 'question' | 'ground_truth',
    value: string,
  ) => {
    setQuestions(
      questions.map((item) =>
        item.key === key ? { ...item, [field]: value } : item,
      ),
    );
  };

  const handleFileChange: UploadProps['customRequest'] = ({ file }) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;

      const processData = (data: any[]) => {
        if (!Array.isArray(data)) {
          message.error(formatMessage({ id: 'tips.upload.error' }));
          return;
        }

        const existingQuestions = new Set(
          questions.map((q) => `${q.question.trim()}|||${q.ground_truth.trim()}`),
        );

        const uniqueNewQuestions: QuestionItem[] = [];
        let tempNextKey = nextKey;

        data.forEach((item) => {
          const question = item.question?.trim();
          const ground_truth = item.ground_truth?.trim();

          if (!question || !ground_truth) {
            return; // 2. 校验：跳过 question 或 ground_truth 为空的数据
          }

          const combined = `${question}|||${ground_truth}`;
          if (!existingQuestions.has(combined)) {
            existingQuestions.add(combined); // 3. 去重：添加到 set 以便后续文件内去重
            uniqueNewQuestions.push({
              key: tempNextKey,
              question,
              ground_truth,
            });
            tempNextKey++;
          }
        });

        if (uniqueNewQuestions.length > 0) {
          setQuestions([...questions, ...uniqueNewQuestions]);
          setNextKey(tempNextKey);
          message.success(
            formatMessage(
              { id: 'tips.import.success' },
              { count: uniqueNewQuestions.length },
            ),
          );
        } else {
          message.info(formatMessage({ id: 'tips.upload.nodata' }));
        }
      };

      if ((file as File).type === 'application/json') {
        try {
          const data = JSON.parse(text);
          processData(data);
        } catch (error) {
          message.error(formatMessage({ id: 'tips.upload.error' }));
        }
      } else {
        Papa.parse(text, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            processData(results.data as any[]);
          },
          error: () => {
            message.error(formatMessage({ id: 'tips.upload.error' }));
          },
        });
      }
    };
    reader.readAsText(file as Blob);
  };

  const handleGenerateQuestions = (generatedQuestions: any[]) => {
    const existingQuestions = new Set(
      questions.map((q) => `${q.question.trim()}|||${q.ground_truth.trim()}`),
    );

    const uniqueNewQuestions: QuestionItem[] = [];
    let tempNextKey = nextKey;

    generatedQuestions.forEach((item) => {
      const question = item.question_text?.trim();
      const ground_truth = item.ground_truth?.trim();

      if (!question || !ground_truth) {
        return;
      }

      const combined = `${question}|||${ground_truth}`;
      if (!existingQuestions.has(combined)) {
        existingQuestions.add(combined);
        uniqueNewQuestions.push({
          key: tempNextKey,
          question,
          ground_truth,
        });
        tempNextKey++;
      }
    });

    if (uniqueNewQuestions.length > 0) {
      setQuestions([...questions, ...uniqueNewQuestions]);
      setNextKey(tempNextKey);
      message.success(
        formatMessage(
          { id: 'tips.import.success' },
          { count: uniqueNewQuestions.length },
        ),
      );
    } else {
      message.info(formatMessage({ id: 'tips.upload.nodata' }));
    }
    setIsModalOpen(false);
  };

  const onFinish = (values: any) => {
    const finalQuestions = questions.filter(
      (q) => q.question.trim() && q.ground_truth.trim(),
    );

    if (finalQuestions.length === 0) {
      message.error(formatMessage({ id: 'evaluation.question_sets.empty' }));
      return;
    }

    const questionSet = {
      ...values,
      questions: finalQuestions.map(({ question, ground_truth }) => ({
        question_text: question,
        ground_truth: ground_truth,
      })),
    };
    evaluationApi
      .createQuestionSetApiV1QuestionSetsPost({
        questionSetCreate: questionSet,
      })
      .then((res) => {
        message.success(formatMessage({ id: 'tips.create.success' }));
        refreshQuestionSets();
        const newQuestionSet = res.data;
        if (newQuestionSet.id) {
          history.push(`/evaluations/question-sets/${newQuestionSet.id}`);
        }
      })
      .catch(() => {
        message.error(formatMessage({ id: 'tips.create.failed' }));
      });
  };

  const demoQ1 = formatMessage({
    id: 'evaluation.question_sets.upload_hint.demo.q1',
  });
  const demoA1 = formatMessage({
    id: 'evaluation.question_sets.upload_hint.demo.a1',
  });
  const demoQ2 = formatMessage({
    id: 'evaluation.question_sets.upload_hint.demo.q2',
  });
  const demoA2 = formatMessage({
    id: 'evaluation.question_sets.upload_hint.demo.a2',
  });

  const csvContent = `question,ground_truth
"${demoQ1}","${demoA1}"
"${demoQ2}","${demoA2}"`;

  const jsonContent = `[
  {
    "question": "${demoQ1}",
    "ground_truth": "${demoA1}"
  },
  {
    "question": "${demoQ2}",
    "ground_truth": "${demoA2}"
  }
]`;

  const fileUploadHint = (
    <div style={{ maxWidth: 400 }}>
      <p>
        <b>CSV:</b>
      </p>
      <pre
        style={{
          background: '#f5f5f5',
          padding: '10px',
          borderRadius: '4px',
        }}
      >
        <code>{csvContent}</code>
      </pre>
      <p>
        <b>JSON:</b>
      </p>
      <pre
        style={{
          background: '#f5f5f5',
          padding: '10px',
          borderRadius: '4px',
        }}
      >
        <code>{jsonContent}</code>
      </pre>
    </div>
  );

  return (
    <PageContainer>
      <PageHeader
        title={formatMessage({ id: 'evaluation.question_sets.new' })}
      />
      <Form
        form={form}
        onFinish={onFinish}
        layout="vertical"
        style={{ maxWidth: 800 }}
      >
        <Form.Item
          label={formatMessage({ id: 'text.title' })}
          name="name"
          rules={[
            {
              required: true,
              message: formatMessage({ id: 'text.title.required' }),
            },
          ]}
        >
          <Input />
        </Form.Item>
        <Form.Item
          label={formatMessage({ id: 'text.description' })}
          name="description"
        >
          <TextArea rows={3} />
        </Form.Item>

        <Divider />

        <Flex justify="space-between" align="center">
          <h3>{formatMessage({ id: 'evaluation.question_sets' })}</h3>
          <Space>
            <Popover
              content={fileUploadHint}
              title={formatMessage({
                id: 'evaluation.question_sets.upload_hint.title',
              })}
            >
              <InfoCircleOutlined style={{ color: 'gray' }} />
            </Popover>
            <Upload
              accept=".csv,.json"
              showUploadList={false}
              customRequest={handleFileChange}
            >
              <Button icon={<UploadOutlined />}>
                {formatMessage({
                  id: 'evaluation.question_sets.import_from_file',
                })}
              </Button>
            </Upload>
            <Button icon={<PlusOutlined />} onClick={() => setIsModalOpen(true)}>
              {formatMessage({
                id: 'evaluation.question_sets.generate_from_collection',
              })}
            </Button>
          </Space>
        </Flex>

        <Table
          columns={[
            {
              title: formatMessage({
                id: 'evaluation.question_sets.question',
              }),
              dataIndex: 'question',
              key: 'question',
              width: '45%',
              render: (text, record) => {
                const isEditing =
                  record.key === editingKey && editingField === 'question';
                return isEditing ? (
                  <Input.TextArea
                    ref={inputRef}
                    value={text}
                    onChange={(e) =>
                      handleQuestionChange(
                        record.key,
                        'question',
                        e.target.value,
                      )
                    }
                    onBlur={() => {
                      setEditingKey(null);
                      setEditingField(null);
                    }}
                    autoSize={{ minRows: 1, maxRows: 5 }}
                  />
                ) : (
                  <Tooltip
                    title={formatMessage({ id: 'text.click_to_edit_tooltip' })}
                  >
                    <div
                      style={{
                        minHeight: '32px',
                        padding: '5px 11px',
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word',
                        cursor: 'pointer',
                      }}
                      onClick={() => {
                        setEditingKey(record.key);
                        setEditingField('question');
                      }}
                    >
                      {text || (
                        <span style={{ color: '#ccc' }}>
                          {formatMessage({ id: 'text.click_to_edit' })}
                        </span>
                      )}
                    </div>
                  </Tooltip>
                );
              },
            },
            {
              title: formatMessage({
                id: 'evaluation.question_sets.ground_truth',
              }),
              dataIndex: 'ground_truth',
              key: 'ground_truth',
              width: '45%',
              render: (text, record) => {
                const isEditing =
                  record.key === editingKey && editingField === 'ground_truth';
                return isEditing ? (
                  <Input.TextArea
                    ref={inputRef}
                    value={text}
                    onChange={(e) =>
                      handleQuestionChange(
                        record.key,
                        'ground_truth',
                        e.target.value,
                      )
                    }
                    onBlur={() => {
                      setEditingKey(null);
                      setEditingField(null);
                    }}
                    autoSize={{ minRows: 1, maxRows: 5 }}
                  />
                ) : (
                  <Tooltip
                    title={formatMessage({ id: 'text.click_to_edit_tooltip' })}
                  >
                    <div
                      style={{
                        minHeight: '32px',
                        padding: '5px 11px',
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word',
                        cursor: 'pointer',
                      }}
                      onClick={() => {
                        setEditingKey(record.key);
                        setEditingField('ground_truth');
                      }}
                    >
                      {text || (
                        <span style={{ color: '#ccc' }}>
                          {formatMessage({ id: 'text.click_to_edit' })}
                        </span>
                      )}
                    </div>
                  </Tooltip>
                );
              },
            },
            {
              title: formatMessage({ id: 'common.actions' }),
              key: 'action',
              width: 100,
              render: (_, record) => (
                <Button
                  type="text"
                  danger
                  icon={<DeleteOutlined />}
                  onClick={() => handleDeleteQuestion(record.key)}
                />
              ),
            },
          ]}
          dataSource={questions}
          locale={{
            emptyText: formatMessage({ id: 'text.empty' }),
          }}
          pagination={false}
        />

        <Button
          block
          type="dashed"
          icon={<PlusOutlined />}
          style={{ marginTop: 16 }}
          onClick={handleAddQuestion}
        >
          {formatMessage({ id: 'evaluation.question_sets.add_question' })}
        </Button>

        <Form.Item style={{ marginTop: 32 }}>
          <Button type="primary" htmlType="submit" style={{ marginRight: 8 }}>
            {formatMessage({ id: 'evaluation.new.form.submit' })}
          </Button>
          <Button onClick={() => history.back()}>
            {formatMessage({ id: 'evaluation.cancel' })}
          </Button>
        </Form.Item>
      </Form>
      <QuestionGenerationModal
        open={isModalOpen}
        onCancel={() => setIsModalOpen(false)}
        onGenerated={handleGenerateQuestions}
      />
    </PageContainer>
  );
};

export default NewQuestionSetPage;
