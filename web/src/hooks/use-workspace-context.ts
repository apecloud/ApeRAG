'use client';

import { Bot, Chat, ChatDetails } from '@/api';
import { createContext, useContext } from 'react';

export type WorkspaceContextProps = {
  bot?: Bot;
  chats?: Chat[];
  chatDelete?: (chat: Chat) => void;
  chatCreate?: () => void;
  chatsReload?: () => void;
  chatRename?: (chat: Chat | ChatDetails) => void;
};

export const WorkspaceContext = createContext<WorkspaceContextProps>({});

export const useWorkspaceContext = () => useContext(WorkspaceContext);
