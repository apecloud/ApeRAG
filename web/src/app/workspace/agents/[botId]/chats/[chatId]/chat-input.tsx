import { Collection, ModelSpec } from '@/api';
import { PageContent } from '@/components/page-container';
import { useAgentsContext } from '@/components/providers/agents-provider';
import { Button } from '@/components/ui/button';
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
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';
import _ from 'lodash';
import { Bot, Globe } from 'lucide-react';
import { useLocale, useTranslations } from 'next-intl';
import { useCallback, useEffect, useMemo, useState } from 'react';
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
};

export const ChatInput = ({
  welcome,
  loading,
  disabled,
  onSubmit,
  onCancel,
}: {
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
    };

    setQuery('');
    setSelectedCollections([]);
    onSubmit(data);
  }, [
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
