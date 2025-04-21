import Header from '@/components/Header';
import { TypesModelServiceProviders } from '@/types';
import { useModel, useParams, useIntl } from '@umijs/max';
import ModelServiceProviderForm from './settingsForm';
import { App, Form } from 'antd';

export default () => {
  const { updateModelServiceProvider: updateModelServiceProvider, getModelServiceProvider: getModelServiceProvider, deleteModelServiceProvider: deleteModelServiceProvider } = useModel('modelServiceProvider');
  const { modelServiceProviderName } = useParams();
  const [form] = Form.useForm();
  const modelServiceProvider = getModelServiceProvider(modelServiceProviderName);
  
  if (!modelServiceProvider) return;

  const { modal } = App.useApp();
  const { formatMessage } = useIntl();

  const onDelete = () => {
    modal.confirm({
      title: formatMessage({ id: 'text.delete.help' }),
      content: <>{formatMessage({ id: 'text.model_service_provider' })}: {modelServiceProviderName}<br/>{formatMessage({ id: 'text.delete.confirm' })}</>,
      onOk: async () => {
        await deleteModelServiceProvider(modelServiceProviderName as string);
      },
      okButtonProps: {
        danger: true,
      },
    });
  };
  
  return (
    <div className="workspace">
      <Header
        goback={true}
        title={modelServiceProviderName}
        name="collections"
        page="collections-detele"
        action={onDelete}
      />
      <div className="bd">
      <div className="border-block">
        <ModelServiceProviderForm
          form={form}
          action="edit"
          values={modelServiceProvider}
          onSubmit={(values: TypesModelServiceProviders) => {
            if (!modelServiceProvider.name) return;
            updateModelServiceProvider(modelServiceProvider.name, values);
          }}
        />
      </div>
      </div>
    </div>
  );
};
