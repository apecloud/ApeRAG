/* eslint-disable react/button-has-type */
import CheckedCard from '@/components/CheckedCard';
import {
  COLLECTION_MODEL_OPTIONS_CONFIG,
  getSourceItemByValue,
} from '@/constants';
import { TypesBot } from '@/types';
import { getUniqColor } from '@/utils/color';
import { FormattedMessage, Link, useModel,useIntl,history } from '@umijs/max';
import {
  Card,
  Row,
  Col,
  Divider,
  Form,
  Input,
  InputNumber,
  Switch,
  Radio,
  Select,
  Space,
  Typography,
  theme,
} from 'antd';
import _ from 'lodash';
import { useEffect, useState } from 'react';

type Props = {
  action: 'add' | 'edit';
  values: TypesBot;
  onSubmit: (data: TypesBot) => void;
  form: any;
};

export default ({ onSubmit, action, values, form }: Props) => {

  const { collections, getCollections } = useModel('collection');
  const { models, chractors, getModels, getChractors } = useModel('model');
  const { token } = theme.useToken();
  const [currentModelItem, setCurrentModelItem] = useState<any>();
  const [botType, setBotType] = useState(values?.type||'knowledge');
  const [contextMemory, setContextMemory] = useState(values?.config?.memory);
  const [relatedQuestions, setRelatedQuestions] = useState(values?.config?.use_related_question);
  const [commonPrompt, setCommonPrompt] = useState(values?.config?.llm?.prompt_template||values?.config?.chractor);
  const intl = useIntl();
  const [allChractors, setAllChractors] = useState<any>();

  let currentModel = Form.useWatch(['config', 'model'], form);

  const changeBotType = (event:any)=>{
    const value = event.target.value;
    setBotType(value);
  }

  const changeChractor = (value:any)=>{
    setCommonPrompt(value);
  }

  const onFinish = async () => {
    const data = await form.validateFields();
    console.log(values, data);
    const {id} = values;
    const v = {id, ...data};
    onSubmit({
      ...v,
      collection_ids: _.isString(data.collection_ids)
        ? [data.collection_ids]
        : data.collection_ids,
    });
  };

  const onMemoryChange = (checked:any) => {
    setContextMemory(checked);
  };

  const onUseRQChange = (checked:any) => {
    setRelatedQuestions(checked);
  };

  useEffect(() => {
    if (models && !form.getFieldValue(['config','model'])){
      form.setFieldValue(['config','model'], models[0]?.value);
    };

    setContextMemory(values?.config?.memory);
    form.setFieldValue(['config', 'memory'], values?.config?.memory);

    const model = models.find((m) => m.value === currentModel);
    setCurrentModelItem(model);

    const llm = values.config?.llm;
    let endpoint = llm?.endpoint;
    let template = llm?.prompt_template;
    let memoryTemplate = llm?.memory_prompt_template;
    let contextwindow = _.isNumber(llm?.context_window)?llm?.context_window:3500;
    let temperature = _.isNumber(llm?.temperature)?llm?.temperature:0.3;
    let similaritytopk = _.isNumber(llm?.similarity_topk)?llm?.similarity_topk:3;
    let similarityscorethreshold = _.isNumber(llm?.similarity_score_threshold)?llm?.similarity_score_threshold:0.5;

    if (currentModel !== values.config?.model) {
      setContextMemory(false);
      form.setFieldValue(['config', 'memory'], false);
      // form.setFieldValue(['config', 'llm', 'trial'], false);
      endpoint = model?.endpoint||'';
      template = model?.prompt_template||'';
      // memoryTemplate = model?.memory_prompt_template||'';
      contextwindow = _.isNumber(model?.context_window)?model?.context_window:3500;
      temperature = _.isNumber(model?.temperature)?model?.temperature:0.3;
      similaritytopk = _.isNumber(model?.similarity_topk)?model?.similarity_topk:3;
      similarityscorethreshold = _.isNumber(model?.similarity_score_threshold)?model?.similarity_score_threshold:0.5;
    }

    form.setFieldValue(['config', 'llm', 'endpoint'], endpoint);
    form.setFieldValue(['config', 'llm', 'prompt_template'], botType==='knowledge'?template:commonPrompt);
    // form.setFieldValue(['config', 'llm', 'memory_prompt_template'], botType==='knowledge'?memoryTemplate:commonPrompt);
    form.setFieldValue(['config', 'llm', 'context_window'], contextwindow);
    form.setFieldValue(['config', 'llm', 'temperature'], temperature);
    form.setFieldValue(['config', 'llm', 'similarity_topk'], similaritytopk);
    form.setFieldValue(['config', 'llm', 'similarity_score_threshold'], similarityscorethreshold);

  }, [currentModel, models, botType, commonPrompt]);

  useEffect(()=>{
    if(!collections){
      getCollections(1, null);
    }
    if(!models || !models.length){
      getModels();
    }
    if(!chractors || !chractors.length){
      getChractors();
    }
  },[history.location.pathname])

  useEffect(()=>{
    if(chractors){
      if(!chractors.filter(item=>item.prompt===values.config.chractor).length){
        setAllChractors([{name:'Customized',description:'',prompt:values.config.chractor}, ...chractors])
      }else{
        setAllChractors(chractors);
      }
    }
  },[chractors])

  return (
    <Form
      onFinish={onFinish}
      labelAlign="right"
      colon={false}
      form={form}
      initialValues={{
        ...values,
        collection_ids: values.collections?.map((c) => c.id),
      }}
    >
      <Card bordered={false} style={{ marginBottom: 20, borderRadius: 16 }}>

          <Form.Item
            label={<FormattedMessage id="text.bot.type" />}
            name="type"
            initialValue={botType}
            valuePropName="value"
            className="form-item-wrap"
            required
          >
            <Radio.Group
              onChange={changeBotType}
              buttonStyle="solid">
              <Radio.Button value={"knowledge"}><FormattedMessage id="text.bot.type.knowledge" /></Radio.Button>
              <Radio.Button value={"common"}><FormattedMessage id="text.bot.type.common" /></Radio.Button>
            </Radio.Group>
          </Form.Item>

        { botType==='knowledge' ? (
          <Form.Item
            label={<FormattedMessage id="text.collections" />}
            required
            className="form-item-wrap"
            name="collection_ids"
            rules={[
              {
                required: true,
                message: intl.formatMessage({id:'text.collections'}) + intl.formatMessage({id:'msg.required'}),
              },
            ]}
          >
            <Select
              mode="multiple"
              fieldNames={{
                label: 'label',
              }}
              dropdownRender={(originNode) => {
                return (
                  <div>
                    {originNode}
                    <Divider style={{ marginBlock: 4 }} />
                    <div style={{ textAlign: 'center', paddingBlock: 4 }}>
                      <Link to="/collections/new">
                        <FormattedMessage id="action.collections.add" />
                      </Link>
                    </div>
                  </div>
                );
              }}
              options={collections?.map((collection) => ({
                label: (
                  <Space>
                    <Typography.Text>{collection.title}</Typography.Text>
                    <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                      {getSourceItemByValue(collection.config?.source)?.label}
                    </Typography.Text>
                  </Space>
                ),
                value: collection.id,
              }))}
            />
          </Form.Item>
        ) : (
          <Form.Item
              name={['config','chractor']}
              className="form-item-wrap"
              label={<FormattedMessage id="text.common.character" />}
              rules={[
                {
                  required: true,
                  message: intl.formatMessage({id:'text.common.character'}) + intl.formatMessage({id:'msg.required'}),
                },
              ]}
            >
              <Select
                onChange={changeChractor}
                options={allChractors?.map((item) => ({
                  label: (
                    <Space>
                      <Typography.Text>{item.name}</Typography.Text>
                      <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                        {item.description}
                      </Typography.Text>
                    </Space>
                  ),
                  value: item.prompt,
                }))}
              />
            </Form.Item>
        )}
        <Form.Item
          className="form-item-wrap"
          name="title"
          label={<FormattedMessage id="text.title" />}
          rules={[
            {
              required: true,
              message: <FormattedMessage id="text.title.help" />,
            },
          ]}
        >
          <Input />
        </Form.Item>
        <Form.Item
          className="form-item-wrap textarea-wrap textarea-content"
          name="description"
          label={<FormattedMessage id="text.description" />}
        >
          <Input.TextArea
            maxLength={300}
            showCount
            style={{ resize: 'none', color: token.colorTextSecondary }}
          />
        </Form.Item>
        <Form.Item
          className="model-wrap"
          name={['config', 'model']}
          label={<FormattedMessage id="text.model_name" />}
          rules={[
            {
              required: true,
              message: intl.formatMessage({id:'text.model_name'}) + intl.formatMessage({id:'msg.required'}),
            },
          ]}
        >
          <CheckedCard
            size="large"
            options={models.map((model) => ({
              label: model.label,
              value: model.value,
              icon:
                COLLECTION_MODEL_OPTIONS_CONFIG[model.family_name]?.icon ||
                `https://ui-avatars.com/api/?background=${getUniqColor(
                  model.family_name,
                ).replace('#', '')}&color=fff&name=${model.family_name}`,
              disabled: !model.enabled,
            }))}
          />
        </Form.Item>

        {currentModelItem?.free_tier && (
          <Form.Item
            className="form-item-children-wrap"
            label={<FormattedMessage id="text.trial" />}
            valuePropName="checked"
            name={['config', 'llm', 'trial']}
          >
            <Switch />
          </Form.Item>
        )}
        {currentModel?.indexOf('gpt') !== -1 || currentModel?.indexOf('deepseek') !== -1 || currentModel?.indexOf('glm') !== -1 ? (
          <Form.Item
            className="form-item-children-wrap"
            label={<FormattedMessage id="text.model_config" />}
            style={{ marginBottom: 0 }}
          >
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Form.Item
                  name={['config', 'llm', 'endpoint']}
                >
                  <Input
                    prefix={
                      <Typography.Text type="secondary">
                        Endpoint
                      </Typography.Text>
                    }
                  />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item name={['config', 'llm', 'token']}>
                  <Input
                    prefix={
                      <Typography.Text type="secondary">Token</Typography.Text>
                    }
                  />
                </Form.Item>
              </Col>
            </Row>
          </Form.Item>
        ) : null}

        {['qwen-turbo', 'qwen-plus', 'qwen-max'].includes(
          currentModel,
        ) ? (
          <Form.Item
            className="form-item-children-wrap"
            label={<FormattedMessage id="text.model_config" />}
            name={['config', 'llm', 'api_key']}
          >
            <Input placeholder="API Key" />
          </Form.Item>
        ) : null}

        {['azure-openai'].includes(currentModel) ? (
          <Form.Item
            className="form-item-children-wrap"
            label={<FormattedMessage id="text.model_config" />}
            style={{ marginBottom: 0 }}
          >
            <Row gutter={[8, 0]}>
              <Col span={6}>
                <Form.Item name={['config', 'llm', 'deployment_id']}>
                  <Input placeholder="Deployment ID" />
                </Form.Item>
              </Col>
              <Col span={6}>
                <Form.Item name={['config', 'llm', 'api_version']}>
                  <Input placeholder="API Version" />
                </Form.Item>
              </Col>
              <Col span={6}>
                <Form.Item name={['config', 'llm', 'endpoint']}>
                  <Input placeholder="Endpoint" />
                </Form.Item>
              </Col>
              <Col span={6}>
                <Form.Item name={['config', 'llm', 'token']}>
                  <Input placeholder="Token" />
                </Form.Item>
              </Col>
            </Row>
          </Form.Item>
        ) : null}

        {['baichuan-53b', 'ernie-bot-turbo'].includes(currentModel) ? (
          <Form.Item
            className="form-item-children-wrap"
            label={<FormattedMessage id="text.model_config" />}
            style={{ marginBottom: 0 }}
          >
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Form.Item name={['config', 'llm', 'api_key']}>
                  <Input
                    prefix={
                      <Typography.Text type="secondary">
                        API Key
                      </Typography.Text>
                    }
                  />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item name={['config', 'llm', 'secret_key']}>
                  <Input
                    prefix={
                      <Typography.Text type="secondary">
                        Secret Key
                      </Typography.Text>
                    }
                  />
                </Form.Item>
              </Col>
            </Row>
          </Form.Item>
        ) : null}


        {/* <Form.Item
          className="form-item-children-wrap"
          style={{display:contextMemory?'block':'none'}}
          label={<FormattedMessage id="text.model_prompt_template" />}
          name={['config', 'llm', 'memory_prompt_template']}
        >
          <Input.TextArea
            autoSize={{
              minRows: 8,
              maxRows: 12,
            }}
            style={{ resize: 'none', color: token.colorTextSecondary }}
          />
        </Form.Item> */}

        <Form.Item
          className="form-item-children-wrap"
          // style={{display:contextMemory?'none':'block'}}
          label={<FormattedMessage id="text.model_prompt_template" />}
          name={['config', 'llm', 'prompt_template']}
        >
          <Input.TextArea
            autoSize={{
              minRows: 8,
              maxRows: 12,
            }}
            style={{ resize: 'none', color: token.colorTextSecondary }}
          />
        </Form.Item>


        <Form.Item
          label={<FormattedMessage id="text.model_llm_params" />}
          className="form-item-children-wrap"
        >
          <Row gutter={[16, 16]}>
            <Col span={6}>
              <label className="label-top" title={intl.formatMessage({id:'text.llm.context_window'})}>
                <Form.Item
                  name={['config', 'llm', 'context_window']}
                  rules={[
                    {
                      required: true,
                      message: intl.formatMessage({id:'text.llm.context_window'}) + intl.formatMessage({id:'msg.required'}),
                    },
                  ]}
                >
                  <InputNumber style={{ width: '100%' }} />
                </Form.Item>
              </label>
            </Col>
            <Col span={6}>
              <label className="label-top" title={intl.formatMessage({id:'text.llm.similarity_score_threshold'})}>
                <Form.Item
                  name={['config', 'llm', 'similarity_score_threshold']}
                  rules={[
                    {
                      required: true,
                      message: intl.formatMessage({id:'text.llm.similarity_score_threshold'}) + intl.formatMessage({id:'msg.required'}),
                    },
                  ]}
                >
                  <InputNumber
                    min={0.1}
                    max={1}
                    step={0.1}
                    style={{ width: '100%' }}
                  />
                </Form.Item>
              </label>
            </Col>
            <Col span={6}>
              <label className="label-top" title={intl.formatMessage({id:'text.llm.similarity_topk'})}>
                <Form.Item
                  name={['config', 'llm', 'similarity_topk']}
                  rules={[
                    {
                      required: true,
                      message: intl.formatMessage({id:'text.llm.similarity_topk'}) + intl.formatMessage({id:'msg.required'}),
                    },
                  ]}
                >
                  <InputNumber
                    min={1}
                    max={10}
                    step={1}
                    style={{ width: '100%' }}
                  />
                </Form.Item>
              </label>
            </Col>
            <Col span={6}>
              <label className="label-top" title={intl.formatMessage({id:'text.llm.temperature'})}>
                <Form.Item
                  name={['config', 'llm', 'temperature']}
                  rules={[
                    {
                      required: true,
                      message: intl.formatMessage({id:'text.llm.temperature'}) + intl.formatMessage({id:'msg.required'}),
                    },
                  ]}
                >
                  <InputNumber
                    min={0}
                    max={1}
                    step={0.1}
                    style={{ width: '100%' }}
                  />
                </Form.Item>
              </label>
            </Col>
          </Row>
        </Form.Item>

        { currentModelItem?.memory ? (
          <>
          <Form.Item
            className="form-item-children-wrap"
            label={<><FormattedMessage id="text.model_memory" /><Typography.Text type="secondary">&nbsp;&nbsp;( <FormattedMessage id="text.model_memory_help" /> )</Typography.Text></>}
            valuePropName="checked"
            name={['config', 'memory']}
          >
            <Switch onChange={onMemoryChange} />
          </Form.Item>
          { contextMemory ? (
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Form.Item
                  rules={[{ required:true, type:'integer', message: intl.formatMessage({id:'msg.need_integer'}), transform(value){return Number(value)>0?Number(value):'error'}}]}
                  name={['config', 'memory_length']}
                >
                  <InputNumber min={1}
                    addonBefore={
                      <Typography.Text type="secondary">
                        {intl.formatMessage({id:'text.model_memory_max_token'})}
                      </Typography.Text>
                    }
                  />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item
                  rules={[{ required:true, type:'integer', message: intl.formatMessage({id:'msg.need_integer'}), transform(value){return Number(value)>0?Number(value):'error'}}]}
                  name={['config', 'memory_count']}
                >
                  <InputNumber min={1}
                    addonBefore={
                      <Typography.Text type="secondary">
                        {intl.formatMessage({id:'text.model_memory_max_session'})}
                      </Typography.Text>
                    }
                  />
                </Form.Item>
              </Col>
            </Row>
          ) : null}
        </>
        ) : null}


        { botType==='knowledge' ? (
          <>
          <Form.Item
            className="form-item-children-wrap"
            label={<><FormattedMessage id="text.model_recall" /><Typography.Text type="secondary">&nbsp;&nbsp;( <FormattedMessage id="text.model_recall_help" /> )</Typography.Text></>}
            valuePropName="checked"
            name={['config', 'llm', 'enable_keyword_recall']}
          >
            <Switch />
          </Form.Item>

          <Form.Item
            className="form-item-children-wrap"
            label={<><FormattedMessage id="text.use_related_question" /></>}
            valuePropName="checked"
            name={['config', 'use_related_question']}
          >
            <Switch onChange={onUseRQChange} />
          </Form.Item>
          { relatedQuestions ? (
            <>
            <Form.Item
              className="form-item-wrap"
              name={['config', 'related_question_prompt']}
              label={<FormattedMessage id="text.related_question_tips" />}
            >
              <Input placeholder={intl.formatMessage({id:"text.related_question_prompt"})} />
            </Form.Item>

            <Form.Item
              className="form-item-children-wrap"
              // style={{display:contextMemory?'none':'block'}}
              label={<FormattedMessage id="text.related_prompt_template" />}
              name={['config', 'llm', 'related_prompt_template']}
            >
              <Input.TextArea
                autoSize={{
                  minRows: 8,
                  maxRows: 12,
                }}
                style={{ resize: 'none', color: token.colorTextSecondary }}
              />
            </Form.Item>
            </>
          ) : null}
          </>
        ) : null}

        {action !== 'add' && (
          <button htmltype='submit' style={{ marginTop: '24px' }}>
            <FormattedMessage id="action.update" />
          </button>
        )}
      </Card>
    </Form>
  );
};
