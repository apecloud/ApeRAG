import { BotCreate } from '@/api';

import {
  CheckCard,
  PageContainer,
  PageHeader,
  RefreshButton,
} from '@/components';
import { BOT_TYPE_ICON } from '@/constants';
import { api } from '@/services';
import { PlusOutlined, SearchOutlined } from '@ant-design/icons';
import {
  Avatar,
  Button,
  Card,
  Col,
  Divider,
  Form,
  Input,
  Modal,
  Result,
  Row,
  Space,
  theme,
  Tooltip,
  Typography,
} from 'antd';
import _ from 'lodash';
import moment from 'moment';
import { useEffect, useMemo, useState } from 'react';
import { UndrawEmpty } from 'react-undraw-illustrations';
import { FormattedMessage, history, Link, useIntl, useModel } from 'umi';

type BotSearchParams = {
  title?: string;
};

export default () => {
  const [searchParams, setSearchParams] = useState<BotSearchParams>();
  const [createVisible, setCreateVisible] = useState<boolean>(false);
  const { token } = theme.useToken();
  const { bots, botsLoading, getBots } = useModel('bot');
  const { formatMessage } = useIntl();

  const [form] = Form.useForm<BotCreate>();

  const { setLoading } = useModel('global');

  const onNewBot = () => {
    form.setFieldsValue({
      title: '',
      description: '',
      type: 'knowledge',
    });
    setCreateVisible(true);
  };

  const onCreateBot = async () => {
    setLoading(true);
    const values = await form.validateFields();
    const botRes = await api.botsPost({
      botCreate: values,
    });
    setLoading(false);
    if (botRes.data.id) {
      await api.botsBotIdChatsPost({
        botId: botRes.data.id,
        chatCreate: { title: '' },
      });
      setCreateVisible(false);

      if (values.type === 'agent') {
        history.push(`/bots/${botRes.data.id}/chats`);
      } else {
        history.push(`/bots/${botRes.data.id}/flow`);
      }
    }
  };

  const _bots = useMemo(
    () =>
      bots?.filter((item) => {
        const titleMatch = searchParams?.title
          ? item.title?.includes(searchParams.title)
          : true;
        return titleMatch;
      }),
    [bots, searchParams],
  );

  const header = useMemo(
    () => (
      <PageHeader
        title={formatMessage({ id: 'bot.name' })}
        description={formatMessage({ id: 'bot.description' })}
      >
        <Space>
          <Input
            placeholder={formatMessage({ id: 'action.search' })}
            prefix={
              <Typography.Text disabled>
                <SearchOutlined />
              </Typography.Text>
            }
            onChange={(e) => {
              setSearchParams({
                ...searchParams,
                title: e.currentTarget.value,
              });
            }}
            allowClear
            value={searchParams?.title}
          />

          <Tooltip title={<FormattedMessage id="bot.add" />}>
            <Button onClick={onNewBot} type="primary" icon={<PlusOutlined />} />
          </Tooltip>
          <RefreshButton loading={botsLoading} onClick={() => getBots()} />
        </Space>
      </PageHeader>
    ),
    [searchParams, botsLoading],
  );

  useEffect(() => {
    getBots();
  }, []);

  if (bots === undefined) return;

  return (
    <PageContainer>
      {header}
      {_.isEmpty(_bots) ? (
        <Result
          icon={
            <UndrawEmpty primaryColor={token.colorPrimary} height="200px" />
          }
          subTitle={<FormattedMessage id="text.empty" />}
        />
      ) : (
        <Row gutter={[24, 24]}>
          {_bots?.map((bot) => {
            return (
              <Col key={bot.id} xs={24} sm={12} md={8} lg={6} xl={6} xxl={6}>
                <Link to={`/bots/${bot.id}/chats`}>
                  <Tooltip title={bot.description}>
                    <Card hoverable size="small">
                      <div
                        style={{
                          display: 'flex',
                          gap: 8,
                          alignItems: 'center',
                        }}
                      >
                        <Avatar
                          style={{ flex: 'none' }}
                          size={40}
                          src={bot.type && BOT_TYPE_ICON[bot.type]}
                          shape="square"
                        />
                        <div style={{ flex: 'auto', maxWidth: '75%' }}>
                          <Typography.Text ellipsis strong>
                            {bot.title}
                          </Typography.Text>
                        </div>
                      </div>
                      <Divider style={{ marginBlock: 8 }} />
                      <div
                        style={{
                          display: 'flex',
                          gap: 8,
                          justifyContent: 'space-between',
                          alignItems: 'center',
                        }}
                      >
                        <Typography.Text
                          ellipsis
                          type="secondary"
                          style={{ fontSize: '0.9em', width: '60%' }}
                        >
                          <FormattedMessage id={`bot.type_${bot.type}`} />
                        </Typography.Text>
                        <Typography.Text
                          ellipsis
                          type="secondary"
                          style={{
                            fontSize: '0.9em',
                            width: '40%',
                            textAlign: 'right',
                          }}
                        >
                          {moment(bot.created).fromNow()}
                        </Typography.Text>
                      </div>
                    </Card>
                  </Tooltip>
                </Link>
              </Col>
            );
          })}
        </Row>
      )}

      <Modal
        title={<FormattedMessage id="bot.add" />}
        open={createVisible}
        onCancel={() => setCreateVisible(false)}
        onOk={() => onCreateBot()}
      >
        <Form
          autoComplete="off"
          layout="vertical"
          form={form}
          style={{ marginTop: 20 }}
        >
          <Form.Item
            name="title"
            label={<FormattedMessage id="text.title" />}
            rules={[
              {
                required: true,
                message: <FormattedMessage id="text.title.required" />,
              },
            ]}
          >
            <Input />
          </Form.Item>
          <Form.Item
            label={<FormattedMessage id="bot.type" />}
            name="type"
            valuePropName="value"
            required
          >
            <CheckCard
              layout={{
                span: 12,
              }}
              gutter={[8, 8]}
              options={[
                {
                  label: <FormattedMessage id="bot.type_knowledge" />,
                  icon: BOT_TYPE_ICON['knowledge'],
                  value: 'knowledge',
                },
                {
                  label: <FormattedMessage id="bot.type_common" />,
                  icon: BOT_TYPE_ICON['common'],
                  value: 'common',
                },
                {
                  label: <FormattedMessage id="bot.type_agent" />,
                  icon: BOT_TYPE_ICON['agent'],
                  value: 'agent',
                },
              ]}
            />
          </Form.Item>
          <Form.Item
            name="description"
            label={<FormattedMessage id="text.description" />}
          >
            <Input.TextArea rows={3} />
          </Form.Item>
        </Form>
      </Modal>
    </PageContainer>
  );
};
