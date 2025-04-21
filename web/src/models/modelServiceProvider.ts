import { GetSupportedModelServiceProviders, GetModelServiceProviders, UpdateModelServiceProvider, DeleteModelServiceProvider} from '@/services/modelServiceProvider';
import { TypesModelServiceProviders } from '@/types';
import { App } from 'antd';
import _ from 'lodash';
import { useState } from 'react';
import { history } from '@umijs/max';

export default () => {
  const [supportedModelServiceProviders, setSupportedModelServiceProviders] = useState<TypesModelServiceProviders[]>([]);
  const [modelServiceProviders, setModelServiceProviders] = useState<TypesModelServiceProviders[]>([]);
  const { message } = App.useApp();

  const getSupportedModelServiceProviders = async () => {
    const { data } = await GetSupportedModelServiceProviders();
    setSupportedModelServiceProviders(data);
  };

  const getModelServiceProviders = async () => {
    const { data } = await GetModelServiceProviders();
    setModelServiceProviders(data);
  };

  const getModelServiceProvider = (name?: string): TypesModelServiceProviders | undefined => {
    if (!name) return;
    return modelServiceProviders?.find((c) => String(c.name) === String(name));
  };

  const updateModelServiceProvider = async (provider: string, params: TypesModelServiceProviders) => {
    const { code, data } = await UpdateModelServiceProvider(provider, params);

    if (code === '200' && data?.label) {
      message.success('Update completed');
    }
  };

  const deleteModelServiceProvider = async (provider: string) => {
    const { code } = await DeleteModelServiceProvider(provider);
    if (code === '200') {
      message.success('Delete completed');
      setModelServiceProviders(modelServiceProviders.filter((msp) => msp.name !== provider));
      setTimeout(() => history.push(`/modelServiceProviders`));
    }
  };

  return {
    supportedModelServiceProviders: supportedModelServiceProviders,
    getSupportedModelServiceProviders: getSupportedModelServiceProviders,
    modelServiceProviders: modelServiceProviders,
    getModelServiceProviders: getModelServiceProviders,
    getModelServiceProvider: getModelServiceProvider,
    updateModelServiceProvider: updateModelServiceProvider,
    deleteModelServiceProvider: deleteModelServiceProvider,
  };
};
