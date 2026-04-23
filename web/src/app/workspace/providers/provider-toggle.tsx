import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import { updateProvider } from '@/features/providers/client-api';
import type { Provider } from '@/features/providers/types';
import { DialogDescription } from '@radix-ui/react-dialog';
import { useTranslations } from 'next-intl';
import { useRouter } from 'next/navigation';
import { useCallback, useState } from 'react';
import { toast } from 'sonner';

export const ProviderToggle = ({ provider }: { provider: Provider }) => {
  const [enabledVisible, setEnabledVisible] = useState<boolean>(false);
  const [disabledVisible, setDisabledVisible] = useState<boolean>(false);
  const [apiKey, setApiKey] = useState<string>(provider.api_key || '');
  const page_models = useTranslations('page_models');
  const common_action = useTranslations('common.action');
  const common_tips = useTranslations('common.tips');
  const router = useRouter();

  const handleEnabled = useCallback(async () => {
    if (!apiKey) {
      toast.error('Please enter the api key for the model provider.');
      return;
    }
    await updateProvider(provider.name, {
      label: provider.label,
      base_url: provider.base_url,
      completion_dialect: provider.completion_dialect,
      embedding_dialect: provider.embedding_dialect,
      rerank_dialect: provider.rerank_dialect,
      allow_custom_base_url: provider.allow_custom_base_url,
      extra: provider.extra ?? undefined,
      api_key: apiKey,
      status: 'enable',
    });
    setEnabledVisible(false);
    setTimeout(router.refresh, 300);
  }, [apiKey, provider, router]);

  const handleDisabled = useCallback(async () => {
    await updateProvider(provider.name, {
      label: provider.label,
      base_url: provider.base_url,
      completion_dialect: provider.completion_dialect,
      embedding_dialect: provider.embedding_dialect,
      rerank_dialect: provider.rerank_dialect,
      allow_custom_base_url: provider.allow_custom_base_url,
      extra: provider.extra ?? undefined,
      status: 'disable',
    });
    setDisabledVisible(false);
    setTimeout(router.refresh, 300);
  }, [provider, router]);

  return (
    <>
      <Switch
        checked={Boolean(provider.api_key)}
        onCheckedChange={(checked) => {
          if (!checked) {
            setDisabledVisible(true);
          } else {
            setEnabledVisible(true);
          }
        }}
        className="cursor-pointer"
      />
      <Dialog
        open={enabledVisible}
        onOpenChange={() => setEnabledVisible(false)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{page_models('provider.api_key')}</DialogTitle>
          </DialogHeader>
          <div>
            <Textarea
              value={apiKey}
              onChange={(e) => setApiKey(e.currentTarget.value)}
              placeholder={page_models('provider.api_key_placeholder')}
              className="w-115 resize-none"
            />

            <div className="text-muted-foreground mt-2 text-sm">
              {page_models('provider.api_key_description')}
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEnabledVisible(false)}>
              {common_action('cancel')}
            </Button>
            <Button onClick={handleEnabled}>{common_action('continue')}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={disabledVisible}
        onOpenChange={() => setDisabledVisible(false)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{common_tips('confirm')}</DialogTitle>
            <DialogDescription>
              {page_models('provider.confirm_disabled')}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDisabledVisible(false)}>
              {common_action('cancel')}
            </Button>
            <Button variant="destructive" onClick={handleDisabled}>
              {common_action('continue')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};
