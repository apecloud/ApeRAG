import { Chat } from '@/api';
import { Navbar, NavbarBody, NavbarHeader } from '@/components';
import { api } from '@/services';
import {
  DeleteOutlined,
  EditOutlined,
  MoreOutlined,
  PlusOutlined,
} from '@ant-design/icons';
import {
  Button,
  Divider,
  Dropdown,
  Form,
  Input,
  Menu,
  MenuProps,
  Modal,
  Space,
  Tooltip,
} from 'antd';
import FormItem from 'antd/es/form/FormItem';
import _ from 'lodash';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { history, useIntl, useLocation, useModel, useParams } from 'umi';

type MenuItem = Required<MenuProps>['items'][number];

export const NavbarBot = () => {
  const { botId, chatId } = useParams();
  const [renameVisible, setRenameVisible] = useState<boolean>();
  const { 
    bot, 
    chats, 
    chatsPagination,
    getBot, 
    setBot, 
    setChats, 
    getChats,
    setChatsPagination
  } = useModel('bot');
  const { loading } = useModel('global');

  const [chatLoading, setChatLoading] = useState<boolean>(false);

  const location = useLocation();
  const { formatMessage } = useIntl();
  const [modal, contextHolder] = Modal.useModal();
  const [form] = Form.useForm<Chat>();
  const loadMoreRef = useRef<HTMLDivElement>(null);
  const navbarBodyRef = useRef<HTMLDivElement>(null);

  const onCreateChat = useCallback(async () => {
    if (botId) {
      setChatLoading(true);
      const res = await api.botsBotIdChatsPost({
        botId,
        chatCreate: {
          title: '',
        },
      });
      setChatLoading(false);
      if (res.status === 200 && res?.data.id) {
        setChats((cs) => cs?.concat(res.data));
        history.push(`/bots/${botId}/chats/${res.data.id}`);
      }
    }
  }, [botId]);

  const onRenameChat = useCallback(async () => {
    const data = await form.validateFields();
    if (!data?.id || !data?.bot_id) return;

    setChatLoading(true);
    await api.botsBotIdChatsChatIdPut({
      botId: data.bot_id,
      chatId: data.id,
      chatUpdate: data,
    });
    setChatLoading(false);

    await getChats(data.bot_id);

    setRenameVisible(false);
  }, []);

  const onDeleteChat = useCallback(
    async (item: Chat) => {
      if (!botId || !item.id) return;
      const confirmed = await modal.confirm({
        title: formatMessage({ id: 'action.confirm' }),
        content: formatMessage({ id: 'chat.delete.confirm' }),
        okButtonProps: {
          danger: true,
          loading,
        },
      });
      if (!confirmed) return;
      setChatLoading(true);
      const res = await api.botsBotIdChatsChatIdDelete({
        botId,
        chatId: item.id,
      });
      setChatLoading(false);
      if (res.status === 204) {
        const data = chats?.filter((c) => c.id !== item.id) || [];
        setChats(data);

        if (_.isEmpty(data)) {
          history.push(`/bots/${botId}/chats`);
        } else if (chatId === item.id) {
          history.push(`/bots/${botId}/chats/${data[0]?.id}`);
        }
      }
    },
    [botId, chatId, chats],
  );

  const onLoadMoreChats = useCallback(async () => {
    if (botId && chatsPagination.current * chatsPagination.pageSize < chatsPagination.total) {
      setChatLoading(true);
      const nextPage = chatsPagination.current + 1;
      const res = await api.botsBotIdChatsGet({ 
        botId,
        page: nextPage,
        pageSize: chatsPagination.pageSize,
      });
      setChatLoading(false);
      
      if ((res.data as any).items) {
        setChats((prevChats) => [...(prevChats || []), ...(res.data as any).items]);
        setChatsPagination({
          ...chatsPagination,
          current: nextPage,
        });
      }
    }
  }, [botId, chatsPagination, setChats, setChatsPagination]);

  // Infinite scroll effect
  useEffect(() => {
    const loadMoreElement = loadMoreRef.current;
    const navbarBodyElement = navbarBodyRef.current;
    
    if (!loadMoreElement || !navbarBodyElement) {
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const [entry] = entries;
        // Check if the load more element is visible and we have more data to load
        if (
          entry.isIntersecting && 
          !chatLoading && 
          chatsPagination.current * chatsPagination.pageSize < chatsPagination.total
        ) {
          onLoadMoreChats();
        }
      },
      {
        root: navbarBodyElement,
        rootMargin: '20px',
        threshold: 0.1,
      }
    );

    observer.observe(loadMoreElement);

    return () => {
      observer.disconnect();
    };
  }, [chatLoading, chatsPagination, onLoadMoreChats]);

  const chatMenuItems = useMemo((): MenuItem[] => {
    const chatItems = (chats || []).map((item) => {
      const path = `/bots/${botId}/chats/${item.id}`;
      return {
        style: { paddingRight: 40 },
        label: (
          <>
            {_.truncate(item.title || formatMessage({ id: 'chat.new' }), {
              length: 22,
            })}
            <Dropdown
              menu={{
                items: [
                  {
                    key: 'rename',
                    label: formatMessage({ id: 'action.rename' }),
                    icon: <EditOutlined />,
                    onClick: ({ domEvent }) => {
                      domEvent.stopPropagation();
                      form.setFieldsValue(item);
                      setRenameVisible(true);
                    },
                  },
                  {
                    key: 'delete',
                    label: formatMessage({ id: 'chat.delete' }),
                    icon: <DeleteOutlined />,
                    danger: true,
                    onClick: ({ domEvent }) => {
                      domEvent.stopPropagation();
                      onDeleteChat(item);
                    },
                  },
                ],
              }}
              overlayStyle={{
                width: 160,
              }}
            >
              <Button
                type="text"
                style={{
                  position: 'absolute',
                  top: 4,
                  right: 4,
                }}
                onClick={(e) => {
                  e.stopPropagation();
                }}
                icon={<MoreOutlined />}
              />
            </Dropdown>
          </>
        ),
        key: path,
      };
    });

    // Add invisible load more trigger for infinite scroll
    const hasMoreChats = chatsPagination.current * chatsPagination.pageSize < chatsPagination.total;
    if (hasMoreChats) {
      chatItems.push({
        key: 'load-more-trigger',
        style: { paddingRight: 0 },
        label: (
          <div
            ref={loadMoreRef}
            style={{ 
              height: '1px', 
              width: '100%', 
              background: 'transparent',
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              fontSize: '12px',
              color: '#999',
              padding: '8px 0'
            }}
          >
            {chatLoading ? formatMessage({ id: 'common.loading' }) + '...' : ''}
          </div>
        ),
      });
    }

    return [
      {
        key: 'chat',
        label: (
          <Space
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            {formatMessage({ id: 'chat.all' })} {chatsPagination.total > 0 && `(${chatsPagination.total})`}
            <Tooltip
              title={formatMessage({ id: 'chat.new' })}
              placement="right"
            >
              <Button
                type="text"
                icon={<PlusOutlined />}
                loading={chatLoading}
                onClick={() => onCreateChat()}
                style={{
                  transform: 'translateX(8px)',
                }}
              />
            </Tooltip>
          </Space>
        ),
        type: 'group',
        children: chatItems,
      },
    ];
  }, [botId, chats, chatsPagination, chatLoading, onCreateChat, onDeleteChat, onRenameChat, onLoadMoreChats]);

  useEffect(() => {
    if (botId) {
      getBot(botId);
      getChats(botId);
    }
    return () => {
      setBot(undefined);
      setChats(undefined);
    };
  }, [botId]);

  if (!bot) return;

  return (
    <>
      {contextHolder}
      <Navbar>
        <NavbarHeader
          title={bot?.title || formatMessage({ id: 'bot.name' })}
          backTo="/bots"
        ></NavbarHeader>
        <NavbarBody>
          <div ref={navbarBodyRef} style={{ height: '100%', overflow: 'auto' }}>
            <Menu
              onClick={({ key }) => history.push(key)}
              mode="inline"
              selectedKeys={[location.pathname]}
              items={chatMenuItems}
              style={{
                padding: 0,
                background: 'none',
                border: 'none',
              }}
            />
            {bot.type !== 'agent' && (
              <Menu
                onClick={({ key }) => history.push(key)}
                mode="inline"
                selectedKeys={[location.pathname]}
                items={[
                  {
                    label: formatMessage({ id: 'action.settings' }),
                    key: `/bots/${botId}/settings`,
                    type: 'group',
                    children: [
                      {
                        label: formatMessage({ id: 'flow.settings' }),
                        key: `/bots/${botId}/flow`,
                      },
                    ],
                  },
                ]}
                style={{
                  padding: 0,
                  background: 'none',
                  border: 'none',
                }}
              />
            )}
          </div>
        </NavbarBody>
      </Navbar>

      <Modal
        width={400}
        open={renameVisible}
        onOk={() => onRenameChat()}
        title={formatMessage({ id: 'action.rename' })}
        onCancel={() => setRenameVisible(false)}
        okButtonProps={{
          loading: chatLoading,
        }}
      >
        <Divider />
        <Form autoComplete="off" layout="vertical" form={form}>
          <FormItem name="id" hidden>
            <Input />
          </FormItem>
          <FormItem name="bot_id" hidden>
            <Input />
          </FormItem>
          <FormItem
            required
            name="title"
            label={formatMessage({ id: 'chat.title' })}
            rules={[
              {
                required: true,
                message: formatMessage({ id: 'chat.title_required' }),
              },
            ]}
          >
            <Input />
          </FormItem>
        </Form>
      </Modal>
    </>
  );
};
