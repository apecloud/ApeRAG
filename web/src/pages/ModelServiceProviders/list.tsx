import { useIntl, useModel, history, FormattedMessage} from '@umijs/max';
import { useEffect } from 'react';
import Header from '../../components/Header';
import ModelServiceProviderList from './List/index';
import MSPForm from './form';
import { Form, Button } from 'antd';
import './index.less';
import { TypesModelServiceProviders } from '@/types';

export default () => {
  const { modelServiceProviders: modelServiceProviders, getModelServiceProviders: getModelServiceProviders, updateModelServiceProvider: updateModelServiceProvider} = useModel('modelServiceProvider');

  const modelServiceProvidersDict = modelServiceProviders.reduce<Record<string, TypesModelServiceProviders>>(
    (dict, item) => {
      dict[item.name] = item;
      return dict;
    },
    {}
  );

  const { formatMessage } = useIntl();

  const headerName = formatMessage({
    id: 'nav.model_service_providers',
  });

  const [form] = Form.useForm();

  useEffect(() => {
    getModelServiceProviders();
  }, [history.location.pathname]);
  
  const submitMSPForm = async ()=>{
    await form.validateFields().then((res)=>{
      form.submit();
    },(e)=>{});
  };

  return (
    <div className="workspace">
      <Header
        title={headerName}
      />

    <div className='bd'>
        <div className='content'>
          <ModelServiceProviderList type="table" list={modelServiceProviders} />
          <div className='border-block'>
            <MSPForm
              action="edit"
              form={form}
              values={{
                mspDict: modelServiceProvidersDict
              }}
              onSubmit={(values: TypesModelServiceProviders) => {
                updateModelServiceProvider(values.name, values);
                setTimeout(() => {
                  getModelServiceProviders();
                }, 1000);
              }}
            />
            <Button
              type="primary"
              onClick={submitMSPForm}
              className="floating-update-button"
            >
              <FormattedMessage id="action.update" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};
