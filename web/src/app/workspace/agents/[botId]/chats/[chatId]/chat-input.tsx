import { Collection, ModelSpec } from '@/api';
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
import { Textarea } from '@/components/ui/textarea';
import { Toggle } from '@/components/ui/toggle';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { apiClient } from '@/lib/api/client';
import { cn } from '@/lib/utils';
import _ from 'lodash';
import { Globe, LoaderCircle } from 'lucide-react';
import { useLocale } from 'next-intl';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { BiSolidRightArrow } from 'react-icons/bi';
import { PiStopFill } from 'react-icons/pi';

import { toast } from 'sonner';

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
  loading,
  disabled,
  onSubmit,
  onCancel,
}: {
  loading: boolean;
  disabled: boolean;
  onSubmit: (params: ChatInputSubmitParams) => void;
  onCancel: () => void;
}) => {
  const [isComposing, setIsComposing] = useState<boolean>(false);
  const [collections, setCollections] = useState<Collection[]>([]);
  const [providerModels, setProviderModels] = useState<
    {
      label?: string;
      name?: string;
      models?: ModelSpec[];
    }[]
  >();
  const locale = useLocale();
  const [query, setQuery] = useState<string>('');
  const [selectedCollections, setSelectedCollections] = useState<string[]>([]);

  const [webSearchEnabled, setWebSearchEnabled] = useState(false);
  const [modelName, setModelName] = useState<string>();

  const loadData = useCallback(async () => {
    const [modelRes, collectionsRes] = await Promise.all([
      apiClient.defaultApi.availableModelsPost({
        tagFilterRequest: {
          tag_filters: [{ operation: 'OR', tags: ['enable_for_agent'] }],
        },
      }),
      apiClient.defaultApi.collectionsGet(),
    ]);

    const items = modelRes.data.items?.map((m) => {
      return {
        label: m.label,
        name: m.name,
        models: m.completion,
      };
    });
    setCollections(collectionsRes.data.items || []);
    setProviderModels(items);
  }, []);

  const handleSendMessage = useCallback(() => {
    const _query = _.trim(query);
    if (_.isEmpty(_query) || isComposing) return;

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

    // setQuery('');
    // setSelectedCollections([]);
    onSubmit(data);
  }, [
    collections,
    isComposing,
    locale,
    modelName,
    onSubmit,
    providerModels,
    query,
    selectedCollections,
    webSearchEnabled,
  ]);

  useEffect(() => {
    if (!modelName && providerModels) {
      providerModels.forEach((provider) => {
        provider.models?.forEach((m) => {
          if (m.tags?.some((t) => t === 'default_for_agent_completion')) {
            setModelName(m.model);
          }
        });
      });
    }
  }, [modelName, providerModels]);

  const enabledColelctions = useMemo(() => {
    return collections.filter((c) => !selectedCollections.includes(c.id || ''));
  }, [collections, selectedCollections]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  return (
    <div className="relative flex flex-col gap-2">
      <Label>
        <Mention
          trigger="@"
          className="w-full"
          value={selectedCollections}
          inputValue={query}
          onInputValueChange={setQuery}
          onValueChange={setSelectedCollections}
          onCompositionStart={() => setIsComposing(true)}
          onCompositionEnd={() => setIsComposing(false)}
          onKeyDown={(e) => {
            if (e.key == 'Enter' && e.shiftKey) {
              handleSendMessage();
              e.preventDefault();
            }
          }}
        >
          <MentionInput asChild>
            <Textarea
              className="resize-none rounded-xl pb-15"
              placeholder={
                disabled
                  ? 'Network connection in progress, please wait...'
                  : 'Type @ to mention a collection...'
              }
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
                No collection was found.
              </div>
            )}
          </MentionContent>
        </Mention>

        <div className="absolute bottom-0 flex w-full flex-row items-center justify-between p-2">
          <div></div>
          <div className="flex gap-2">
            <Tooltip>
              <TooltipTrigger asChild>
                <Toggle
                  variant={webSearchEnabled ? 'outline' : 'default'}
                  onClick={() => setWebSearchEnabled(!webSearchEnabled)}
                  aria-label="Web search"
                  className={cn('relative cursor-pointer')}
                  disabled={disabled}
                >
                  <Globe
                    className={`${webSearchEnabled ? 'text-primary' : 'text-muted-foreground'}`}
                  />
                </Toggle>
              </TooltipTrigger>
              <TooltipContent>Web search</TooltipContent>
            </Tooltip>

            <Select
              value={modelName}
              disabled={disabled}
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
              {loading && (
                <LoaderCircle className="absolute size-full animate-spin opacity-30" />
              )}
              {loading ? <PiStopFill /> : <BiSolidRightArrow />}
            </Button>
          </div>
        </div>
      </Label>
    </div>
  );
};
