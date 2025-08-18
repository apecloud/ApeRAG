import { PageContainer } from '@/components/page-container';
import { PageHeader } from '@/components/page-header';
import {
  DeleteOutlined,
  InfoCircleOutlined,
  PlusOutlined,
  UploadOutlined,
} from '@ant-design/icons';
import { QuestionType } from '@/api';
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
import { history, useIntl, useParams, useModel } from 'umi';
import { QuestionGenerationModal } from '../_question_generation_modal';
import { useRequest } from 'ahooks';

const { TextArea } = Input;

interface QuestionItem {
  id?: string; // Existing questions will have an ID
  key: number;
  question: string;
  ground_truth: string;
  question_type?: QuestionType;
}

const QuestionSetDetailPage = () => {
  const { formatMessage } = useIntl();
  const { message } = App.useApp();
  const { refresh: refreshQuestionSets } = useModel('questionSet');
  const [form] = Form.useForm();
  const params = useParams<{ questionSetId: string }>();
  const qsId = params.questionSetId;

  // State for questions displayed in the table
  const [questions, setQuestions] = useState<QuestionItem[]>([]);

  // State to track changes
  const [addedQuestions, setAddedQuestions] = useState<QuestionItem[]>([]);
  const [updatedQuestions, setUpdatedQuestions] = useState<QuestionItem[]>([]);
  const [deletedQuestionIds, setDeletedQuestionIds] = useState<string[]>([]);

  const [nextKey, setNextKey] = useState(1);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [editingKey, setEditingKey] = useState<number | null>(null);
  const [editingField, setEditingField] = useState<string | null>(null);
  const inputRef = useRef<any>(null);
  const evaluationApi = new EvaluationApi();

  const { loading: loadingQuestionSet, refresh } = useRequest(
    () => {
      if (!qsId) return Promise.resolve(null);
      return evaluationApi.getQuestionSetApiV1QuestionSetsQsIdGet({ qsId });
    },
    {
      refreshDeps: [qsId],
      onSuccess: (res) => {
        if (res) {
          form.setFieldsValue(res.data);
          const initialQuestions =
            res.data.questions?.map((q, i) => ({
              id: q.id,
              key: i + 1,
              question: q.question_text || '',
              ground_truth: q.ground_truth || '',
              question_type: q.question_type,
            })) || [];
          setQuestions(initialQuestions);
          setNextKey(initialQuestions.length + 1);
        }
      },
    },
  );

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
      question_type: QuestionType.USER_DEFINED,
    };
    setQuestions([...questions, newQuestion]);
    setAddedQuestions([...addedQuestions, newQuestion]);
    setNextKey(newKey + 1);
    setEditingKey(newKey);
    setEditingField('question');
  };

  const handleDeleteQuestion = (key: number, id?: string) => {
    setQuestions(questions.filter((item) => item.key !== key));
    if (id) {
      // If it's an existing question, add its ID to the deleted list
      setDeletedQuestionIds([...deletedQuestionIds, id]);
    } else {
      // If it's a newly added question, just remove it from the added list
      setAddedQuestions(addedQuestions.filter((item) => item.key !== key));
    }
  };

  const handleQuestionChange = (
    key: number,
    field: 'question' | 'ground_truth',
    value: string,
  ) => {
    const newQuestions = questions.map((item) =>
      item.key === key ? { ...item, [field]: value } : item,
    );
    setQuestions(newQuestions);

    const changedItem = newQuestions.find((item) => item.key === key);
    if (changedItem) {
      if (changedItem.id) {
        // It's an existing question, add to updated list
        const otherUpdates = updatedQuestions.filter(q => q.id !== changedItem.id);
        setUpdatedQuestions([...otherUpdates, changedItem]);
      } else {
        // It's a new question, update it in the added list
        const otherAdds = addedQuestions.filter(q => q.key !== changedItem.key);
        setAddedQuestions([...otherAdds, changedItem]);
      }
    }
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

        let tempNextKey = nextKey;
        const newQuestionsFromFile: QuestionItem[] = [];

        data.forEach((item) => {
          const question = item.question?.trim();
          const ground_truth = item.ground_truth?.trim();

          if (!question || !ground_truth) {
            return;
          }

          const combined = `${question}|||${ground_truth}`;
          if (!existingQuestions.has(combined)) {
            existingQuestions.add(combined);
            const newQ = {
              key: tempNextKey,
              question,
              ground_truth,
            };
            newQuestionsFromFile.push(newQ);
            tempNextKey++;
          }
        });

        if (newQuestionsFromFile.length > 0) {
          setQuestions([...questions, ...newQuestionsFromFile]);
          setAddedQuestions([...addedQuestions, ...newQuestionsFromFile]);
          setNextKey(tempNextKey);
          message.success(
            formatMessage(
              { id: 'tips.import.success' },
              { count: newQuestionsFromFile.length },
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

    let tempNextKey = nextKey;
    const newQuestionsFromGeneration: QuestionItem[] = [];

    generatedQuestions.forEach((item) => {
      const question = item.question_text?.trim();
      const ground_truth = item.ground_truth?.trim();

      if (!question || !ground_truth) {
        return;
      }

      const combined = `${question}|||${ground_truth}`;
      if (!existingQuestions.has(combined)) {
        existingQuestions.add(combined);
        const newQ = {
          key: tempNextKey,
          question,
          ground_truth,
        };
        newQuestionsFromGeneration.push(newQ);
        tempNextKey++;
      }
    });

    if (newQuestionsFromGeneration.length > 0) {
      setQuestions([...questions, ...newQuestionsFromGeneration]);
      setAddedQuestions([...addedQuestions, ...newQuestionsFromGeneration]);
      setNextKey(tempNextKey);
      message.success(
        formatMessage(
          { id: 'tips.import.success' },
          { count: newQuestionsFromGeneration.length },
        ),
      );
    } else {
      message.info(formatMessage({ id: 'tips.upload.nodata' }));
    }
    setIsModalOpen(false);
  };

  const onFinish = async (values: any) => {
    if (!qsId) return;

    try {
      // 1. Update question set info (name, description)
      await evaluationApi.updateQuestionSetApiV1QuestionSetsQsIdPut({
        qsId,
        questionSetUpdate: { name: values.name, description: values.description },
      });

      // 2. Delete questions
      const deletePromises = deletedQuestionIds.map(id =>
        evaluationApi.deleteQuestionApiV1QuestionSetsQsIdQuestionsQIdDelete({ qsId, qId: id })
      );

      // 3. Update questions
      const updatePromises = updatedQuestions.map((q) =>
        evaluationApi.updateQuestionApiV1QuestionSetsQsIdQuestionsQIdPut({
          qsId,
          qId: q.id!,
          questionUpdate: {
            question_text: q.question,
            ground_truth: q.ground_truth,
            question_type: q.question_type,
          },
        }),
      );

      // 4. Add new questions
      let addPromise: Promise<any> = Promise.resolve();
      if (addedQuestions.length > 0) {
        const questionsToAdd = addedQuestions.map(q => ({
          question_text: q.question,
          ground_truth: q.ground_truth,
          question_type: q.question_type,
        }));
        addPromise = evaluationApi.addQuestionsApiV1QuestionSetsQsIdQuestionsPost({
          qsId,
          questionsAdd: { questions: questionsToAdd },
        });
      }


      await Promise.all([...deletePromises, ...updatePromises, addPromise]);

      message.success(formatMessage({ id: 'tips.update.success' }));
      // Reset tracking states and refresh data
      setAddedQuestions([]);
      setUpdatedQuestions([]);
      setDeletedQuestionIds([]);
      refresh();
      refreshQuestionSets();
      history.push(`/evaluations/question-sets/${qsId}`);
    } catch (error) {
      message.error(formatMessage({ id: 'tips.update.failed' }));
    }
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
        title={formatMessage({ id: 'evaluation.question_sets.edit' })}
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
          loading={loadingQuestionSet}
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
                  onClick={() => handleDeleteQuestion(record.key, record.id)}
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
          <Space>
            <Button type="primary" htmlType="submit">
              {formatMessage({ id: 'evaluation.question_sets.update' })}
            </Button>
            <Button onClick={() => history.push(`/evaluations/question-sets/${qsId}`)}>
              {formatMessage({ id: 'evaluation.cancel' })}
            </Button>
          </Space>
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

export default QuestionSetDetailPage;
