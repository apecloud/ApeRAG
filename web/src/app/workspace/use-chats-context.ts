'use client';

import { Bot, Chat, ChatDetails } from '@/api';
import { createContext, useContext } from 'react';

export type ChatsContextProps = {
  bot?: Bot;
  chats?: Chat[];
  chatDelete?: (chat: Chat) => void;
  chatCreate?: () => void;
  chatsReload?: () => void;
  chatRename?: (chat: Chat | ChatDetails) => void;
};

export const ChatsContext = createContext<ChatsContextProps>({});

export const useChatsContext = () => useContext(ChatsContext);
