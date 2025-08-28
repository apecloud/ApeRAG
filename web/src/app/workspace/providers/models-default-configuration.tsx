import {
  DefaultModelConfig,
  DefaultModelConfigScenarioEnum,
  ModelSpec,
} from '@/api';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { apiClient } from '@/lib/api/client';
import _ from 'lodash';
import { Settings, X } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { toast } from 'sonner';

export const ModelsDefaultConfiguration = () => {
  const [defaultModels, setDefaultModels] = useState<DefaultModelConfig[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [visible, setVisible] = useState<boolean>(false);
  const [scenarioModels, setScenarioModels] = useState<{
    [key in DefaultModelConfigScenarioEnum]: {
      label?: string;
      name?: string;
      models?: ModelSpec[];
    }[];
  }>();

  const loadModels = useCallback(async () => {
    setLoading(true);
    const [defaultModelsRes, collectionModelsRes, agentModelsRes] =
      await Promise.all([
        apiClient.defaultApi.defaultModelsGet(),
        apiClient.defaultApi.availableModelsPost({
          tagFilterRequest: {
            tag_filters: [
              { operation: 'AND', tags: ['enable_for_collection'] },
            ],
          },
        }),
        apiClient.defaultApi.availableModelsPost({
          tagFilterRequest: {
            tag_filters: [{ operation: 'AND', tags: ['enable_for_agent'] }],
          },
        }),
      ]);
    setLoading(false);
    setDefaultModels(defaultModelsRes.data.items || []);

    const agentModels = agentModelsRes.data.items || [];
    const collectionModels = collectionModelsRes.data.items || [];

    const default_for_agent_completion = agentModels.map((m) => ({
      label: m.label,
      name: m.name,
      models: m.completion,
    }));
    const default_for_collection_completion = collectionModels.map((m) => ({
      label: m.label,
      name: m.name,
      models: m.completion,
    }));
    const default_for_embedding = collectionModels.map((m) => ({
      label: m.label,
      name: m.name,
      models: m.embedding,
    }));
    const default_for_rerank = collectionModels.map((m) => ({
      label: m.label,
      name: m.name,
      models: m.rerank,
    }));
    const default_for_background_task = agentModels.map((m) => ({
      label: m.label,
      name: m.name,
      models: m.completion,
    }));

    setScenarioModels({
      default_for_agent_completion,
      default_for_collection_completion,
      default_for_embedding,
      default_for_rerank,
      default_for_background_task,
    });
  }, []);

  const handleScenarioChange = useCallback(
    (scenario: DefaultModelConfigScenarioEnum, model?: string) => {
      setDefaultModels((items) => {
        const item = items.find((m) => m.scenario === scenario);
        if (item) {
          item.model = model;
          item.provider_name = scenarioModels?.[scenario].find((s) =>
            s.models?.some((m) => m.model === model),
          )?.name;
        }
        return [...items];
      });
    },
    [scenarioModels],
  );

  const handleSave = useCallback(async () => {
    const res = await apiClient.defaultApi.defaultModelsPut({
      defaultModelsUpdateRequest: { defaults: defaultModels },
    });
    if (res?.status === 200) {
      setVisible(false);
      toast.success('update successfully.');
    }
  }, [defaultModels]);

  const content = useMemo(() => {
    if (loading) {
      return (
        <>
          {_.times(5).map((index) => {
            return (
              <div key={index} className="flex w-full flex-col gap-2">
                <Skeleton className="h-[14px] w-1/2 rounded-md" />
                <Skeleton className="h-[36px] w-full rounded-md" />
              </div>
            );
          })}
          <Skeleton className="h-[40px] w-full rounded-md" />
        </>
      );
    } else {
      return (
        <>
          {defaultModels.map((modelConfig) => {
            return (
              <div
                key={modelConfig.scenario}
                className="flex w-full flex-col gap-2"
              >
                <Label>{_.startCase(modelConfig.scenario)}</Label>
                <div className="flex flex-row gap-1">
                  <Select
                    value={
                      defaultModels.find(
                        (m) => m.scenario === modelConfig.scenario,
                      )?.model || undefined
                    }
                    onValueChange={(v) => {
                      handleScenarioChange(modelConfig.scenario, v);
                    }}
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="Select a model" />
                    </SelectTrigger>
                    <SelectContent>
                      {scenarioModels?.[modelConfig.scenario]
                        .filter((item) => _.size(item.models))
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
                    variant="outline"
                    onClick={() => {
                      handleScenarioChange(modelConfig.scenario, undefined);
                    }}
                  >
                    <X />
                  </Button>
                </div>
              </div>
            );
          })}
          <div className="text-muted-foreground text-sm">
            Clearing selection will delete the default model configuration for
            this scenario
          </div>
        </>
      );
    }
  }, [defaultModels, handleScenarioChange, loading, scenarioModels]);

  useEffect(() => {
    if (visible) {
      loadModels();
    }
  }, [loadModels, visible]);

  return (
    <>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button variant="outline" onClick={() => setVisible(true)}>
            <Settings />
          </Button>
        </TooltipTrigger>
        <TooltipContent>Default models configuration</TooltipContent>
      </Tooltip>
      <Dialog open={visible} onOpenChange={() => setVisible(false)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Default models configuration</DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-6 py-8">{content}</div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setVisible(false)}>
              Cancel
            </Button>
            <Button onClick={handleSave}>Save</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};
