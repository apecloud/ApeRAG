import { Switch } from '@/components/ui/switch';
import { updateProviderModelTags } from '@/features/providers/client-api';
import type { Provider, ProviderModel } from '@/features/providers/types';
import _ from 'lodash';
import { useTranslations } from 'next-intl';
import { useRouter } from 'next/navigation';
import { useCallback } from 'react';
import { toast } from 'sonner';

export const ModelTagSwitch = ({
  model,
  provider,
  tag,
}: {
  model: ProviderModel;
  provider: Provider;
  tag: string;
}) => {
  const common_tips = useTranslations('common.tips');
  const router = useRouter();
  const handleTagChange = useCallback(
    async (checked: boolean) => {
      const tags = checked
        ? (model.tags || []).concat(tag)
        : (model.tags || []).filter((t) => t !== tag);

      await updateProviderModelTags(provider.name, model, _.uniq(tags));
      setTimeout(router.refresh, 300);
      toast.success(common_tips('update_success'));
    },
    [common_tips, model, provider.name, router.refresh, tag],
  );

  return (
    <Switch
      onCheckedChange={handleTagChange}
      checked={model.tags?.includes(tag)}
      className="cursor-pointer"
    />
  );
};
