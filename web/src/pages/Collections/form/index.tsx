import CheckedCard from '@/components/CheckedCard';
import { DOCUMENT_SOURCE_OPTIONS, EMAIL_CONNECT_INFO, COLLECTION_MODEL_SERVICE_PROVIDER_OPTIONS_CONFIG } from '@/constants';
import { TypesCollection, TypesEmailSource } from '@/types';
import {
  Alert,
  Avatar,
  Card,
  Form,
  Input,
  Switch,
  Radio,
  Select,
  Segmented,
  Space,
  Typography,
} from 'antd';
import { useEffect, useState } from 'react';
import DocumentEmailFormItems from './DocumentEmailFormItems';
import DocumentFeishuFormItems from './DocumentFeishuFormItems';
import DocumentFtpFormItems from './DocumentFtpFormItems';
import DocumentLocalFormItems from './DocumentLocalFormItems';
import DocumentGithubFormItems from './DocumentGithubFormItems';
import IconEmail from '@/assets/email.svg';
import IconGmail from '@/assets/gmail.ico';
import IconOutlook from '@/assets/outlook.png';
import IconQQ from '@/assets/qq.ico';
import { useIntl,useModel } from '@umijs/max';
import DocumentCloudFormItems from './DocumentCloudFormItems';

type Props = {
  action: 'add' | 'edit';
  values: TypesCollection;
  form: any;
  onSubmit: (data: TypesCollection) => void;
};

export default ({ onSubmit, action, values, form }: Props) => {
  const { user } = useModel('user');
  const { embeddings, getEmbeddings} = useModel('embedding');
  const [desciption, setDesciption] = useState('');
  const [sensitiveProtect, setSensitiveProtect] = useState(values?.config?.sensitive_protect);
  const [ readonly, setReadonly ] = useState(values?.system && !user?.is_admin);
  const { formatMessage } = useIntl();

  const source = Form.useWatch(['config', 'source'], form);
  const emailSource: TypesEmailSource | undefined = Form.useWatch(
    ['config', 'email_source'],
    form,
  );
  const embeddingModel = Form.useWatch(['config', 'embedding_model'], form);

  const onFinish = async () => {
    const data = form.getFieldsValue();
    let buckets = data.config?.buckets;
    if(buckets?.length>1){
      buckets = [...new Set(buckets.map((item)=>JSON.stringify(item).toLowerCase()))].map(JSON.parse);
      data.config.buckets = buckets;
    }
    onSubmit(data);
  };

  const onSPChange = (checked) => {
    setSensitiveProtect(checked);
  };

  useEffect(() => {
    getEmbeddings();
  }, []);

  useEffect(() => {
    form.setFieldsValue(values);
  }, []);

  useEffect(() => {
    if (source === 'ftp') {
      form.setFieldValue(['config', 'port'], 21);
    }
    if (source === 'email') {
      if (!emailSource && action !== 'edit') {
        form.setFieldValue(['config', 'email_source'], 'gmail');
      }
      if (emailSource) {
        form.setFieldValue(
          ['config', 'pop_server'],
          EMAIL_CONNECT_INFO[emailSource].pop_server,
        );
        form.setFieldValue(
          ['config', 'port'],
          EMAIL_CONNECT_INFO[emailSource].port,
        );
      }
    }
  }, [source, emailSource]);

  useEffect(() => {
    if(embeddingModel) {
      const [model_service_provider, embedding_name] = embeddingModel.split(':');
      form.setFieldValue(['config', 'embedding_model_service_provider'], model_service_provider);
      form.setFieldValue(["config", "embedding_model_name"], embedding_name);
    }
  }, [embeddingModel]);

  return (
    <Form
      onFinish={onFinish}
      labelAlign="right"
      colon={false}
      disabled={readonly}
      initialValues={{
        config: {
          enable_light_rag: true,
        }
      }}
      form={form}
    >
      <Card bordered={false} style={{ marginBottom: 20 }}>
        <Form.Item
          name="title"
          label={formatMessage({id:"text.title"})}
          rules={[
            {
              required: true,
              message: formatMessage({id:"text.title"}) + formatMessage({id:"msg.required"}),
            },
          ]}
          className="form-item-wrap"
        >
          <Input />
        </Form.Item>
        <Form.Item
          name="description"
          label={formatMessage({id:"text.description"})}
          className="form-item-wrap textarea-content"
        >
          <div className="textarea-content">
            <Input.TextArea
              maxLength={300}
              autoSize={{ minRows: 1, maxRows: 3 }}
              defaultValue={values.description}
              onChange={(e) => setDesciption(e.target.value)}
            />
            <span className="count">{desciption.length}/300</span>
          </div>
        </Form.Item>

        <Form.Item
          className="form-item-children-wrap"
          label={<><Typography.Text>{formatMessage({id:"text.sensitive.protect"})}</Typography.Text>&nbsp;<Typography.Text type="secondary">({formatMessage({id:"text.sensitive_help"})})</Typography.Text></>}
          valuePropName="checked"
          name={['config','sensitive_protect']}
        >
          <Switch onChange={onSPChange} />
        </Form.Item>
        { sensitiveProtect ? (
          <>
          {/* <Form.Item
            className="form-item-children-wrap"
            label={<><Typography.Text>{formatMessage({id:"text.sensitive.protect_llm"})}</Typography.Text>&nbsp;<Typography.Text type="secondary">({formatMessage({id:"text.sensitive.protect_llm_help"})})</Typography.Text></>}
            valuePropName="checked"
            name={['config','sensitive_protect_llm']}
          >
            <Switch />
          </Form.Item> */}
          <Form.Item 
            valuePropName="value"
            name={['config','sensitive_protect_method']}
          >
            <Radio.Group defaultValue={'nostore'} buttonStyle="solid">
              <Radio.Button value="nostore">{formatMessage({id:"text.sensitive.nostore"})}</Radio.Button>
              <Radio.Button value="replace">{formatMessage({id:"text.sensitive.replace"})}</Radio.Button>
            </Radio.Group>
          </Form.Item>
          </>
        ) : null}

        {/* <Form.Item
          className="form-item-children-wrap"
          label={<Typography.Text>{formatMessage({id:"text.use_related_question"})}</Typography.Text>}
          valuePropName="checked"
          name={['config','enable_related_questions']}
        >
          <Switch />
        </Form.Item> */}

        <Form.Item
          name="type"
          style={{ display: 'none' }}
          className="form-item-wrap"
        >
          <Input />
        </Form.Item>
        <Form.Item
          name={['config', 'source']}
          rules={[
            {
              required: true,
              message: formatMessage({id:"text.source"}) + formatMessage({id:"msg.required"}),
            },
          ]}
          className="form-item-wrap"
          label={formatMessage({id:"text.source"})}
        >
          <CheckedCard
            size="large"
            disabled={action === 'edit'}
            options={DOCUMENT_SOURCE_OPTIONS.map((s) => ({
              label: s.label,
              value: s.value,
              icon: source === s.value ? s.icon_on : s.icon,
            }))}
          />
        </Form.Item>

        {source === 'local' ? DocumentLocalFormItems(formatMessage) : null}
        {source === 'email' ? (
          <>
            <Form.Item label="" name={['config', 'email_source']}>
              <Segmented
                size="small"
                block
                options={[
                  {
                    label: (
                      <Space size="small" style={{ padding: 4 }}>
                        <Avatar src={IconGmail} size={24} />
                        <Typography.Text>Gmail</Typography.Text>
                      </Space>
                    ),
                    value: 'gmail',
                  },
                  {
                    label: (
                      <Space size="small" style={{ padding: 4 }}>
                        <Avatar src={IconOutlook} size={24} />
                        <Typography.Text>Outlook</Typography.Text>
                      </Space>
                    ),
                    value: 'outlook',
                  },
                  {
                    label: (
                      <Space size="small" style={{ padding: 4 }}>
                        <Avatar src={IconQQ} size={24} />
                        <Typography.Text>QQMail</Typography.Text>
                      </Space>
                    ),
                    value: 'qqmail',
                  },
                  {
                    label: (
                      <Space size="small" style={{ padding: 4 }}>
                        <Avatar src={IconEmail} size={24} />
                        <Typography.Text>Others</Typography.Text>
                      </Space>
                    ),
                    value: 'others',
                  },
                ]}
              />
            </Form.Item>
            {DocumentEmailFormItems(formatMessage)}
            {emailSource ? (
              <Form.Item label="">
                <Alert
                  message={EMAIL_CONNECT_INFO[emailSource].title}
                  description={
                    <>
                      {EMAIL_CONNECT_INFO[emailSource].tips.map(
                        (tip, index) => (
                          <div key={index}>
                            <Typography.Text type="secondary">
                              {index + 1}. {tip}
                            </Typography.Text>
                          </div>
                        ),
                      )}
                    </>
                  }
                  type="info"
                  showIcon
                />
              </Form.Item>
            ) : null}
          </>
        ) : null}
        {source === 's3' || source === 'oss' ? DocumentCloudFormItems(formatMessage) : null}
        {source === 'ftp' ? DocumentFtpFormItems(formatMessage) : null}
        {source === 'feishu' ? DocumentFeishuFormItems(formatMessage) : null}
        {source === 'github' ? DocumentGithubFormItems(formatMessage) : null}

        <Form.Item
          name={['config', 'embedding_model']}
          rules={[
            {
              required: true,
              message: formatMessage({id:"text.embedding_model"}) + formatMessage({id:"msg.required"}),
            },
          ]}
          className="form-item-wrap"
          label={formatMessage({id:"text.embedding_model"})}
        >
            <Select
              disabled={action === 'edit'}
              fieldNames={{
                label: 'label',
              }}
              options={embeddings?.map((embedding) => ({
                label: (
                  <Space>
                    <img
                      src = {COLLECTION_MODEL_SERVICE_PROVIDER_OPTIONS_CONFIG[embedding.model_service_provider]?.icon}
                      alt="Custom Icon"
                      style={{
                        width: 16,
                        height: 16,
                        marginLeft: 8,
                        verticalAlign: 'middle',
                      }}
                    />
                    <Typography.Text>{embedding.model_service_provider}</Typography.Text>
                    <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                      {embedding.embedding_name}
                    </Typography.Text>
                  </Space>
                ),
                value: embedding.model_service_provider + ":" + embedding.embedding_name
              }))}
            />
        </Form.Item>

        <Form.Item name={['config', 'embedding_model_service_provider']} hidden>
          <input type="hidden" />
        </Form.Item>
        <Form.Item name={['config', 'embedding_model_name']} hidden>
          <input type="hidden" />
        </Form.Item>

        <Form.Item
          className="form-item-children-wrap"
          label={formatMessage({id:"text.enable_light_rag"})}
          valuePropName="checked"
          name={['config', 'enable_light_rag']}
        >
          <Switch />
        </Form.Item>

        <Form.Item
          label=" "
          style={{ textAlign: 'left' }}
          className="form-item-wrap"
        >
          {action !== 'add' && (
            // eslint-disable-next-line react/button-has-type
            <button htmltype="submit">
              {formatMessage({id:"action.update"})}
            </button>
          )}
        </Form.Item>

      </Card>
    </Form>
  );
};
