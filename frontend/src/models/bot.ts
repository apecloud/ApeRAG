import { Bot, Chat, ChatDetails } from '@/api';
import { api } from '@/services';
import { parseConfig } from '@/utils';
import { useState } from 'react';
import { useModel } from 'umi';

export default () => {
  const { setLoading } = useModel('global');

  const [bot, setBot] = useState<Bot>();
  const [botLoading, setBotLoading] = useState<boolean>(false);

  const [bots, setBots] = useState<Bot[]>();
  const [botsLoading, setBotsLoading] = useState<boolean>(false);
  const [botsPagination, setBotsPagination] = useState({
    current: 1,
    pageSize: 20,
    total: 0,
  });

  const [chat, setChat] = useState<ChatDetails>();

  const [chats, setChats] = useState<Chat[]>();
  const [chatsLoading, setChatsLoading] = useState<boolean>(false);
  const [chatsPagination, setChatsPagination] = useState({
    current: 1,
    pageSize: 50,
    total: 0,
  });

  // bots
  const getBots = async (page?: number, pageSize?: number) => {
    setLoading(true);
    setBotsLoading(true);
    const res = await api.botsGet({
      page: page || botsPagination.current,
      pageSize: pageSize || botsPagination.pageSize,
    });

    setLoading(false);
    setBotsLoading(false);
    setBots(
      res.data.items?.map((item) => ({
        ...item,
        config: parseConfig(item.config),
      })),
    );
    
    if (res.data.pageResult) {
      setBotsPagination({
        current: res.data.pageResult.page_number || 1,
        pageSize: res.data.pageResult.page_size || 20,
        total: res.data.pageResult.count || 0,
      });
    }
  };

  // bot
  const getBot = async (botId: string) => {
    setLoading(true);
    setBotLoading(true);
    const res = await api.botsBotIdGet({ botId });

    setLoading(false);
    setBotLoading(false);

    setBot(res.data);
  };

  // chats
  const getChats = async (botId: string, page?: number, pageSize?: number) => {
    setLoading(true);
    setChatsLoading(true);
    const res = await api.botsBotIdChatsGet({ 
      botId,
      page: page || chatsPagination.current,
      pageSize: pageSize || chatsPagination.pageSize,
    });
    setLoading(false);
    setChatsLoading(false);
    setChats(res.data.items);
    
    if (res.data.pageResult) {
      setChatsPagination({
        current: res.data.pageResult.page_number || 1,
        pageSize: res.data.pageResult.page_size || 50,
        total: res.data.pageResult.count || 0,
      });
    }
  };

  // chat
  const getChat = async (botId: string, chatId: string) => {
    setLoading(true);
    const res = await api.botsBotIdChatsChatIdGet({
      botId,
      chatId,
    });
    setLoading(false);
    setChat(res.data);
  };

  return {
    bot,
    botLoading,
    getBot,
    setBot,

    bots,
    botsLoading,
    botsPagination,
    getBots,
    setBots,
    setBotsPagination,

    chats,
    chatsLoading,
    chatsPagination,
    getChats,
    setChats,
    setChatsPagination,

    chat,
    getChat,
    setChat,
  };
};
