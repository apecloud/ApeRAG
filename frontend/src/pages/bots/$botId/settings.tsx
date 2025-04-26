import { Bot } from '@/api';
import { PageContainer, PageHeader } from '@/components';
import { api } from '@/services';
import { stringifyConfig } from '@/utils';
import { Card, Form } from 'antd';
import { toast } from 'react-toastify';
import { useIntl, useModel } from 'umi';
import BotForm from '../_form';

export default () => {
  const { formatMessage } = useIntl();
  const { bot, getBot } = useModel('bot');
  const { setLoading } = useModel('global');
  const [form] = Form.useForm<Bot>();
  const onFinish = async (values: Bot) => {
    if (!bot?.id) return;
    setLoading(true);
    const res = await api.botsBotIdPut({
      botId: bot.id,
      botUpdate: {
        ...values,
        config: stringifyConfig(values.config),
      },
    });
    setLoading(false);
    if (res.status === 200) {
      getBot(bot.id);
      toast.success(formatMessage({ id: 'tips.update.success' }));
    }
  };

  if (!bot) return;

  return (
    <PageContainer>
      <PageHeader title={formatMessage({ id: 'bot.settings' })} />

      <Card variant="borderless">
        <BotForm
          form={form}
          onSubmit={(values: Bot) => onFinish(values)}
          action="edit"
          values={bot}
        />
      </Card>
    </PageContainer>
  );
};
