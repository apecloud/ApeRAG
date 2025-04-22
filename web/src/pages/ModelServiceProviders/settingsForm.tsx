import { TypesModelServiceProviders } from '@/types';
import { getUniqColor } from '@/utils/color';
import { FormattedMessage, Link, useModel,useIntl,history } from '@umijs/max';
import {
  Card,
  Form,
  Input,
} from 'antd';
import _ from 'lodash';
import { useEffect, useState } from 'react';

type Props = {
  action: 'add' | 'edit';
  values: TypesModelServiceProviders;
  onSubmit: (data: TypesModelServiceProviders) => void;
  form: any;
};

export default ({ onSubmit, action, values, form }: Props) => {
  const intl = useIntl();

  const onFinish = async () => {
    const data = await form.validateFields();
    console.log(values, data);
    const {name} = values;
    const v = {name, ...data};
    onSubmit({
      ...v
    });
  };

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

       {values["allow_custom_base_url"] && (
        <Form.Item
            className="form-item-children-wrap"
            label={<FormattedMessage id="text.model_service_provider_url" />}
            name={['base_url']}
          >
            <Input placeholder="Base URL" />
          </Form.Item>
       )}

        <Form.Item
            className="form-item-children-wrap"
            label={<FormattedMessage id="text.model_service_provider_apikey" />}
            name={['api_key']}
          >
            <Input placeholder="API Key" />
          </Form.Item>
 
        {action !== 'add' && (
          <button htmltype='submit' style={{ marginTop: '24px' }}>
            <FormattedMessage id="action.update" />
          </button>
        )}
      </Card>
    </Form>
  );
};
