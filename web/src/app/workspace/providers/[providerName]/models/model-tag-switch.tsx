import { LlmProvider, LlmProviderModel } from '@/api';
import { Switch } from '@/components/ui/switch';
import { apiClient } from '@/lib/api/client';
import _ from 'lodash';
import { useRouter } from 'next/navigation';
import { useCallback } from 'react';
import { toast } from 'sonner';

export const ModelTagSwitch = ({
  model,
  provider,
  tag,
}: {
  model: LlmProviderModel;
  provider: LlmProvider;
  tag: string;
}) => {
  const router = useRouter();
  const handleTagChange = useCallback(
    async (checked: boolean) => {
      const tags = checked
        ? (model.tags || []).concat(tag)
        : (model.tags || []).filter((t) => t !== tag);

      const res =
        await apiClient.defaultApi.llmProvidersProviderNameModelsApiModelPut({
          providerName: provider.name,
          api: model.api,
          model: model.model,
          llmProviderModelUpdate: {
            ...model,
            tags: _.uniq(tags),
          },
        });

      if (res?.status === 200) {
        setTimeout(router.refresh, 300);
        toast.success('Saved successfully.');
      }
    },
    [model, provider.name, router.refresh, tag],
  );

  return (
    <Switch
      onCheckedChange={handleTagChange}
      checked={model.tags?.includes(tag)}
      className="cursor-pointer"
    />
  );
};
