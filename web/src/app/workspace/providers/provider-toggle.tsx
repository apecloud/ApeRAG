import { LlmProvider } from '@/api';
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
import { apiClient } from '@/lib/api/client';
import { DialogDescription } from '@radix-ui/react-dialog';
import { useRouter } from 'next/navigation';
import { useCallback, useState } from 'react';
import { toast } from 'sonner';

export const ProviderToggle = ({ provider }: { provider: LlmProvider }) => {
  const [enabledVisible, setEnabledVisible] = useState<boolean>(false);
  const [disabledVisible, setDisabledVisible] = useState<boolean>(false);
  const [apiKey, setApiKey] = useState<string>(provider.api_key || '');

  const router = useRouter();

  const handleEnabled = useCallback(async () => {
    if (!apiKey) {
      toast.error('Please enter the api key for the model provider.');
      return;
    }
    const res = await apiClient.defaultApi.llmProvidersProviderNamePut({
      providerName: provider.name,
      llmProviderUpdateWithApiKey: {
        ...provider,
        api_key: apiKey,
        status: 'enable',
      },
    });
    if (res.data.name) {
      setEnabledVisible(false);
      setTimeout(router.refresh, 300);
    }
  }, [apiKey, provider, router]);

  const handleDisabled = useCallback(async () => {
    const res = await apiClient.defaultApi.llmProvidersProviderNamePut({
      providerName: provider.name,
      llmProviderUpdateWithApiKey: {
        ...provider,
        status: 'disable',
      },
    });
    if (res.data.name) {
      setDisabledVisible(false);
      setTimeout(router.refresh, 300);
    }
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
            <DialogTitle>API Key</DialogTitle>
          </DialogHeader>
          <div>
            <Textarea
              value={apiKey}
              onChange={(e) => setApiKey(e.currentTarget.value)}
              placeholder="Please enter the api key for the model provider."
              className="w-115 resize-none"
            />

            <div className="text-muted-foreground mt-2 text-sm">
              To access the AI model&apos;s capabilities, you need to provide a
              valid API Key from your chosen model provider. This key serves as
              a secure authentication method, allowing the system to process
              your requests while maintaining privacy and usage control.
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEnabledVisible(false)}>
              Cancel
            </Button>
            <Button onClick={handleEnabled}>OK</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={disabledVisible}
        onOpenChange={() => setDisabledVisible(false)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Confirm</DialogTitle>
            <DialogDescription>
              Confirm disabling this Provider?
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDisabledVisible(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDisabled}>
              OK
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};
