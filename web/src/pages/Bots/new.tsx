import { TypesBot } from '@/types';
import { useModel,useParams } from '@umijs/max';
import Header from '../../components/Header';
import BotForm from './form';
import { Form } from 'antd';

export default () => {
  const { botId } = useParams();
  const { getBot,createBot } = useModel('bot');
  const bot = botId?getBot(botId):null;
  const { models } = useModel('model');
  const currentModel = models?.[0];
  const [form] = Form.useForm();
  const submitBotForm = async ()=>{
    await form.validateFields().then((res)=>{
      form.submit();
    },(e)=>{});
  };

  return (
    <div className='workspace'>
      <Header goback={true} title="Add a Bot" page="bots-new" action={submitBotForm} />
      <div className="bd">
        <div className="content">
          <div className='border-block'>
            <BotForm
              action="add"
              form={form}
              values={{
                config: {
                  model: currentModel?.value,
                  llm: {
                    endpoint: currentModel?.endpoint,
                    similarity_score_threshold: currentModel?.similarity_score_threshold,
                    similarity_topk: currentModel?.similarity_topk,
                    temperature: currentModel?.temperature,
                    context_window: currentModel?.context_window,
                    prompt_template: currentModel?.prompt_template,
                    memory_prompt_template: currentModel?.memory_prompt_template,
                  }
                },
              }}
              onSubmit={(values: TypesBot) => {
                createBot(values);
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
