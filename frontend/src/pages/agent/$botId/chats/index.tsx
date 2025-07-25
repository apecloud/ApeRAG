import { PageContainer } from '@/components';
import { api } from '@/services';
import { Button, Result, theme } from 'antd';
import { useCallback, useState } from 'react';
import { UndrawFirmware } from 'react-undraw-illustrations';
import { FormattedMessage, history, useModel, useParams } from 'umi';

export default () => {
  const { chats, setChats } = useModel('bot');
  const { botId } = useParams();
  const { token } = theme.useToken();
  const [botsCreating, setBotsCreating] = useState<boolean>(false);

  const onCreateChat = useCallback(async () => {
    if (botId) {
      setBotsCreating(true);
      const res = await api.botsBotIdChatsPost({
        botId,
        chatCreate: {
          title: '',
        },
      });
      setBotsCreating(false);
      if (res.status === 200 && res?.data.id) {
        setChats((cs) => cs?.concat(res.data));
        history.push(`/bots/${botId}/chats/${res.data.id}`);
      }
    }
  }, [botId]);

  if (chats === undefined) return;

  return (
    <PageContainer>
      <Result
        icon={
          <UndrawFirmware primaryColor={token.colorPrimary} height="200px" />
        }
        subTitle={<FormattedMessage id="chat.empty_description" />}
        extra={
          <Button loading={botsCreating} type="primary" onClick={onCreateChat}>
            <FormattedMessage id="chat.start_new" />
          </Button>
        }
      />
    </PageContainer>
  );
};
