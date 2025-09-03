import {
  ChatDetails,
  Collection,
  ModelSpec,
  UploadDocumentResponseStatusEnum,
} from '@/api';
import { PageContent } from '@/components/page-container';
import { useAgentsContext } from '@/components/providers/agents-provider';
import { Button } from '@/components/ui/button';
import { FileUpload, FileUploadTrigger } from '@/components/ui/file-upload';
import { Label } from '@/components/ui/label';
import {
  Mention,
  MentionContent,
  MentionInput,
  MentionItem,
} from '@/components/ui/mention';
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useSidebar } from '@/components/ui/sidebar';
import { Textarea } from '@/components/ui/textarea';
import { Toggle } from '@/components/ui/toggle';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { apiClient } from '@/lib/api/client';
import { cn } from '@/lib/utils';
import { useInterval } from 'ahooks';
import { motion } from 'framer-motion';
import _ from 'lodash';
import { Bot, Globe, LoaderCircle, Paperclip, Trash2 } from 'lucide-react';
import { useLocale, useTranslations } from 'next-intl';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { defaultStyles, FileIcon } from 'react-file-icon';
import { BiSolidRightArrow } from 'react-icons/bi';
import { PiStopFill } from 'react-icons/pi';
import { toast } from 'sonner';
import useLocalStorageState from 'use-local-storage-state';

export type ChatInputSubmitParams = {
  query: string;
  collections: Collection[];
  completion: {
    model: string;
    model_service_provider: string;
    custom_llm_provider: string;
  };
  web_search_enabled: boolean;
  language: string;
  files: {
    id: string;
    name: string;
  }[];
};

export type Attachment = {
  id: string;
  file: File;
  progress_status:
    | 'pending'
    | 'uploading'
    | 'uploaded'
    | 'indexing'
    | 'success'
    | 'failed';

  document_id?: string;
  filename?: string;
  size?: number;
  status?: UploadDocumentResponseStatusEnum;
};

export const ChatInput = ({
  chat,
  welcome,
  loading,
  disabled,
  onSubmit,
  onCancel,
}: {
  chat: ChatDetails;
  welcome: boolean;
  loading: boolean;
  disabled: boolean;
  onSubmit: (params: ChatInputSubmitParams) => void;
  onCancel: () => void;
}) => {
  const [isComposing, setIsComposing] = useState<boolean>(false);
  const { open, isMobile } = useSidebar();
  const { providerModels, collections } = useAgentsContext();
  const [mentionOpen, setMentionOpen] = useState<boolean>(false);
  const locale = useLocale();
  const [query, setQuery] = useState<string>('');
  const [selectedCollections, setSelectedCollections] = useState<string[]>([]);
  const page_chat = useTranslations('page_chat');
  const [webSearchEnabled, setWebSearchEnabled] = useLocalStorageState<boolean>(
    'web-search-enabled',
    {
      defaultValue: false,
    },
  );
  const [modelName, setModelName] = useLocalStorageState<string | undefined>(
    'local-agent-completion-model',
  );
  const [attachments, setAttachments] = useState<Attachment[]>([]);

  const handleDeleteAttachment = useCallback(async (attachment: Attachment) => {
    setAttachments((items) =>
      items.filter((item) => item.id !== attachment.id),
    );
  }, []);

  const checkAttachmentStatus = useCallback(async () => {
    const chatId = chat.id;
    if (!chatId) return;
    const indexingAttachments = attachments.filter((attachment) => {
      return (
        ['uploaded'].includes(attachment.progress_status) &&
        attachment.document_id
      );
    });
    await Promise.all(
      indexingAttachments.map(async (attachment) => {
        setAttachments((items) => {
          const item = items.find((item) => item.id === attachment.id);
          if (item) {
            item.progress_status = 'indexing';
          }
          return [...items];
        });
        const res =
          await apiClient.chatDocumentsApi.chatsChatIdDocumentsDocumentIdGet({
            chatId,
            documentId: attachment.document_id || '',
          });

        if (res.data) {
          setAttachments((items) => {
            const item = items.find((item) => item.id === attachment.id);
            if (item) {
              switch (res.data.status) {
                case 'COMPLETE':
                  item.progress_status = 'success';
                  break;
                case 'FAILED':
                  item.progress_status = 'failed';
                  break;
                default:
                  item.progress_status = 'indexing';
              }
            }
            return [...items];
          });
        }
      }),
    );
  }, [attachments, chat.id]);

  const onAttachmentsChange = useCallback(async () => {
    const chatId = chat.id;
    if (!chatId) return;

    const uploadAttachments = attachments.filter((attachment) => {
      return (
        ['pending'].includes(attachment.progress_status) &&
        !attachment.document_id
      );
    });
    await Promise.all(
      uploadAttachments.map(async (attachment) => {
        const file = attachment.file;
        setAttachments((items) => {
          const item = items.find((item) => item.id === attachment.id);
          if (item) {
            item.progress_status = 'uploading';
          }
          return [...items];
        });
        const res = await apiClient.chatDocumentsApi.chatsChatIdDocumentsPost({
          chatId,
          file,
          messageId: '',
        });
        if (res.data.id) {
          setAttachments((items) => {
            const item = items.find((item) => item.id === attachment.id);
            if (item) {
              item.document_id = res.data.id;
              item.progress_status = 'uploaded';
            }
            return [...items];
          });
        }
      }),
    );
  }, [attachments, chat.id]);

  const onFileReject = useCallback((file: File, message: string) => {
    toast.error(message, {
      description: `"${file.name.length > 20 ? `${file.name.slice(0, 20)}...` : file.name}" has been rejected`,
    });
  }, []);

  const onFileValidate = useCallback(
    (file: File): string | null => {
      const doc = attachments.some(
        (attachment) =>
          attachment.file.name === file.name &&
          attachment.file.size === file.size &&
          attachment.file.lastModified === file.lastModified &&
          attachment.file.type === file.type,
      );
      if (doc) {
        return 'File already exists.';
      }
      return null;
    },
    [attachments],
  );

  useInterval(() => {
    checkAttachmentStatus();
  }, 3000);

  useEffect(() => {
    onAttachmentsChange();
  }, [attachments, onAttachmentsChange]);

  const handleSendMessage = useCallback(() => {
    const _query = _.trim(query);
    if (_.isEmpty(_query) || isComposing || loading || mentionOpen) return;

    let model: ModelSpec | undefined;
    const provider = providerModels?.find((p) =>
      p.models?.some((m) => m.model === modelName),
    );

    providerModels?.forEach((provider) => {
      provider.models?.forEach((m) => {
        if (m.model === modelName) {
          model = m;
        }
      });
    });

    if (!modelName || model === undefined) {
      toast.error(`Please select an LLM model.`);
      return;
    }

    const data = {
      query: _query,
      collections: collections.filter((c) =>
        selectedCollections.some((id) => c.id === id),
      ),
      completion: {
        model: modelName,
        model_service_provider: provider?.name || '',
        custom_llm_provider: model.custom_llm_provider || '',
      },
      web_search_enabled: webSearchEnabled,
      language: locale,
      files: attachments
        .filter((attachment) => attachment.progress_status === 'success')
        .map((attachment) => ({
          id: attachment.document_id || '',
          name: attachment.file.name,
        })),
    };

    setQuery('');
    setSelectedCollections([]);
    onSubmit(data);
  }, [
    attachments,
    collections,
    isComposing,
    loading,
    locale,
    mentionOpen,
    modelName,
    onSubmit,
    providerModels,
    query,
    selectedCollections,
    webSearchEnabled,
  ]);

  useEffect(() => {
    if (_.isEmpty(providerModels)) {
      return;
    }
    let defaultModel: string | undefined;
    let includesCurrentModel = false;
    providerModels?.forEach((provider) => {
      provider.models?.forEach((m) => {
        if (m.tags?.some((t) => t === 'default_for_agent_completion')) {
          defaultModel = m.model;
        }
        if (m.model === modelName) {
          includesCurrentModel = true;
        }
      });
    });
    if (!includesCurrentModel) {
      setModelName(defaultModel);
    }
  }, [modelName, providerModels, setModelName]);

  const enabledColelctions = useMemo(() => {
    return collections.filter((c) => !selectedCollections.includes(c.id || ''));
  }, [collections, selectedCollections]);

  return (
    <div
      className={cn(
        'bg-background/95 fixed right-0 z-10 backdrop-blur-lg transition-[width,height,left] ease-linear',
        !open || isMobile ? 'left-0' : 'left-[var(--sidebar-width)]',
        welcome ? 'top-[25%]' : 'bottom-0',
      )}
    >
      <PageContent className="xs:px-4 pb-8 sm:px-8 md:px-12 lg:px-20">
        {welcome && (
          <div className="mb-6 flex flex-col justify-center text-center">
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                duration: 0.3,
                ease: 'easeIn',
                delay: 0,
              }}
              className="relative mx-auto mb-6 flex size-18 justify-center"
            >
              <Bot className="size-full opacity-10" />
              <div className="opacity-30">
                <div className="animate-caret-blink absolute top-9 left-6 h-3 w-1.5 rounded-sm bg-black dark:bg-white"></div>
                <div className="animate-caret-blink absolute top-9 left-10.5 h-3 w-1.5 rounded-sm bg-black delay-75 dark:bg-white"></div>
              </div>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                duration: 0.3,
                ease: 'easeIn',
                delay: 0.1,
              }}
              className="mb-2 text-xl font-medium"
            >
              {page_chat('hello_world')}
            </motion.div>
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                duration: 0.3,
                ease: 'easeIn',
                delay: 0.2,
              }}
              className="text-muted-foreground text-sm"
            >
              {page_chat('rag_description')}
            </motion.div>
          </div>
        )}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{
            duration: 0.3,
            ease: 'easeIn',
            delay: 0.3,
          }}
          className="relative flex flex-col gap-2"
        >
          <div className="flex flex-wrap gap-2">
            {attachments.map((attachment) => {
              const extension = _.last(attachment.file.type.split('/')) || '';
              return (
                <div
                  key={attachment.id}
                  className="bg-accent group hover:bg-accent/80 relative flex flex-row items-center gap-1 rounded-md border p-1 text-xs transition-colors"
                >
                  <div className="size-6">
                    {['success', 'failed'].includes(
                      attachment.progress_status,
                    ) ? (
                      <FileIcon
                        color="var(--primary)"
                        extension={extension}
                        {..._.get(defaultStyles, extension)}
                      />
                    ) : (
                      <LoaderCircle className="size-6 animate-spin opacity-50" />
                    )}
                  </div>
                  <div className="flex-1">
                    <div className="w-30 truncate">{attachment.file.name}</div>
                    <div className="text-muted-foreground flex flex-row justify-between">
                      <span>
                        {(attachment.file.size / 1000).toFixed(0) + ' Kb'}
                      </span>
                      <span>{attachment.progress_status}</span>
                    </div>
                  </div>
                  <div
                    onClick={() => handleDeleteAttachment(attachment)}
                    className="bg-accent absolute -top-2 -right-2 z-10 flex size-6 cursor-pointer flex-col justify-center rounded-full p-1 text-center opacity-0 transition-opacity group-hover:opacity-100"
                  >
                    <Trash2 className="m-auto size-3 text-rose-500" />
                  </div>
                </div>
              );
            })}
          </div>

          <Label>
            <Mention
              trigger="@"
              className="w-full"
              open={mentionOpen}
              onOpenChange={setMentionOpen}
              value={selectedCollections}
              inputValue={query}
              onInputValueChange={setQuery}
              onValueChange={setSelectedCollections}
              onCompositionStart={() => setIsComposing(true)}
              onCompositionEnd={() => setIsComposing(false)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  handleSendMessage();
                  e.preventDefault();
                }
              }}
            >
              <MentionInput asChild>
                <Textarea
                  className="resize-none rounded-xl pb-20"
                  value={query}
                  placeholder={page_chat('mention_a_collection')}
                  disabled={disabled}
                />
              </MentionInput>
              <MentionContent className="w-60">
                {enabledColelctions.length ? (
                  enabledColelctions.map((collection) => (
                    <MentionItem
                      key={collection.id}
                      value={collection.id || ''}
                      className="flex-col items-start gap-0.5"
                      disabled={collection.status !== 'ACTIVE'}
                    >
                      <span className="text-sm">{collection.title}</span>
                      <span className="text-muted-foreground text-xs">
                        {collection.id}
                      </span>
                    </MentionItem>
                  ))
                ) : (
                  <div className="text-muted-foreground p-4 text-center text-xs">
                    {page_chat('no_collection_was_found')}
                  </div>
                )}
              </MentionContent>
            </Mention>

            <div className="absolute bottom-0 flex w-full flex-row items-center justify-between p-4">
              <div></div>
              <div className="flex gap-2">
                <FileUpload
                  maxFiles={10}
                  maxSize={10 * 1024 * 1024}
                  accept=".pdf,.doc,.docx,.txt,.md,.ppt,.pptx,.xls,.xlsx"
                  value={attachments.map((f) => f.file)}
                  onFileReject={onFileReject}
                  onFileValidate={onFileValidate}
                  onValueChange={(files) => {
                    setAttachments((attachments) => {
                      const data: Attachment[] = [];
                      files.forEach((file) => {
                        const attachment = attachments.find((attachment) =>
                          _.isEqual(attachment.file, file),
                        );
                        data.push({
                          id: String(Math.random()),
                          file,
                          progress_status: 'pending',
                          ...attachment,
                        });
                      });
                      return data;
                    });
                  }}
                >
                  <FileUploadTrigger asChild>
                    <Button
                      size="icon"
                      variant="ghost"
                      className="cursor-pointer"
                    >
                      <Paperclip />
                    </Button>
                  </FileUploadTrigger>
                </FileUpload>

                <Tooltip>
                  <TooltipTrigger asChild>
                    <Toggle
                      variant={webSearchEnabled ? 'outline' : 'default'}
                      onClick={() => {
                        const enabled = !webSearchEnabled;
                        toast.success(
                          enabled
                            ? page_chat('web_search_is_enabled')
                            : page_chat('web_search_is_disabled'),
                        );
                        setWebSearchEnabled(enabled);
                      }}
                      aria-label={page_chat('web_search')}
                      className={cn('relative cursor-pointer')}
                      disabled={disabled}
                    >
                      <Globe
                        className={`${webSearchEnabled ? 'text-primary' : 'text-muted-foreground'}`}
                      />
                    </Toggle>
                  </TooltipTrigger>
                  <TooltipContent>{page_chat('web_search')}</TooltipContent>
                </Tooltip>

                <Select
                  value={modelName}
                  disabled={disabled}
                  defaultValue={modelName}
                  onValueChange={(v) => {
                    setModelName(v);
                  }}
                >
                  <SelectTrigger className="w-60 cursor-pointer">
                    <SelectValue placeholder="Select a model" />
                  </SelectTrigger>
                  <SelectContent>
                    {providerModels
                      ?.filter((item) => _.size(item.models))
                      .map((item) => {
                        return (
                          <SelectGroup key={item.name}>
                            <SelectLabel>{item.label}</SelectLabel>
                            {item.models?.map((model) => {
                              return (
                                <SelectItem
                                  key={model.model}
                                  value={model.model || ''}
                                >
                                  {model.model}
                                </SelectItem>
                              );
                            })}
                          </SelectGroup>
                        );
                      })}
                  </SelectContent>
                </Select>
                <Button
                  size="icon"
                  disabled={disabled}
                  className={cn('relative cursor-pointer rounded-full')}
                  onClick={() => {
                    if (loading) {
                      onCancel();
                    } else {
                      handleSendMessage();
                    }
                  }}
                >
                  {loading ? <PiStopFill /> : <BiSolidRightArrow />}
                </Button>
              </div>
            </div>
          </Label>
        </motion.div>
      </PageContent>
    </div>
  );
};
