/* eslint-disable react/button-has-type */
import CheckedCard from '@/components/CheckedCard';
import {
  COLLECTION_MODEL_SERVICE_PROVIDER_OPTIONS_CONFIG,
} from '@/constants';
import { TypesModelServiceProviders } from '@/types';
import { getUniqColor } from '@/utils/color';
import { FormattedMessage, Link, useModel,useIntl,history } from '@umijs/max';
import {
  Card,
  Row,
  Col,
  Form,
  Input,
} from 'antd';
import _ from 'lodash';
import { useEffect} from 'react';

type Props = {
  action: 'add' | 'edit';
  values: {
    mspDict: Record<string, TypesModelServiceProviders>;
  }
  onSubmit: (data: TypesModelServiceProviders) => void;
  form: any;
};

export default ({ onSubmit, action, values, form }: Props) => {

  const { supportedModelServiceProviders, getSupportedModelServiceProviders } = useModel('modelServiceProvider');
  const intl = useIntl();

  let currentModel = Form.useWatch(['name'], form);

  const onFinish = async () => {
    const data = await form.validateFields();
    console.log(values, data);
    onSubmit({...data});
  };

  useEffect(()=>{
    if(!supportedModelServiceProviders || !supportedModelServiceProviders.length){
      getSupportedModelServiceProviders();
    }
  },[history.location.pathname])

  return (
    <Form
      onFinish={onFinish}
      labelAlign="right"
      colon={false}
      form={form}
      initialValues={{
        ...values
      }}
    >

      <Card bordered={false} style={{ marginBottom: 20, borderRadius: 16 }}>

        <Form.Item
          className="model-wrap"
          name={['name']}
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
            options={supportedModelServiceProviders.map((msp) => ({
              label: msp.label,
              value: msp.name,
              icon:
              COLLECTION_MODEL_SERVICE_PROVIDER_OPTIONS_CONFIG[msp.name]?.icon ||
                `https://ui-avatars.com/api/?background=${getUniqColor(
                  msp.name,
                ).replace('#', '')}&color=fff&name=${msp.name}`
            }))}
          />
        </Form.Item>

        {['openai'].includes(
          currentModel,
        ) ? (
          <Form.Item
            className="form-item-children-wrap"
            label={<FormattedMessage id="text.model_config" />}
            style={{ marginBottom: 0 }}
          >
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Form.Item
                  name={['base_url']}
                >
                  <Input placeholder = {values.mspDict?.[currentModel]?.base_url || "Endpoint"}/>
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item name={['api_key']}>
                  <Input placeholder = {values.mspDict?.[currentModel]?.api_key || "Token"}/>
                </Form.Item>
              </Col>
            </Row>
          </Form.Item>
        ) : null}

        {['alibabacloud', 'siliconflow', 'deepseek'].includes(
          currentModel,
        ) ? (
          <Form.Item
            className="form-item-children-wrap"
            label={<FormattedMessage id="text.model_config" />}
            name={['api_key']}
          >
            <Input placeholder = {values.mspDict?.[currentModel]?.api_key || "Token"}/>
          </Form.Item>
        ) : null}

      </Card>
    </Form>
  );
};
