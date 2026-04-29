'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import {
  createModel,
  createModelAccount,
  deleteModelAccount,
  isModelAllowedForScenario,
  updateModel,
  updateModelAccount,
  updateModelUse,
  validateModelAccount,
} from '@/features/providers/client-api';
import type {
  Model,
  ModelAccount,
  ModelCapability,
  ModelPlatformViewModel,
  ModelProvider,
  ModelUseScenario,
} from '@/features/providers/types';
import { cn } from '@/lib/utils';
import { Loader2 } from 'lucide-react';
import { useTranslations } from 'next-intl';
import { useRouter } from 'next/navigation';
import { useMemo, useState } from 'react';
import { toast } from 'sonner';

type ModelPreset = {
  providerType: string;
  providerModelId: string;
  displayName: string;
  capability: ModelCapability;
  contextWindow?: number;
  embeddingDimensions?: number;
  supportsVision?: boolean;
  supportsToolCalling?: boolean;
  supportsMultimodalEmbedding?: boolean;
};

type ProviderForm = {
  name: string;
  displayName: string;
  baseUrl: string;
  apiKey: string;
};

type ModelForm = {
  providerModelId: string;
  displayName: string;
  capability: ModelCapability;
};

const capabilityMeta: Record<ModelCapability, { label: string }> = {
  chat: {
    label: '对话',
  },
  completion: {
    label: '补全',
  },
  embedding: {
    label: '向量',
  },
};

const isSupportedModelCapability = (
  capability: string,
): capability is ModelCapability => capability in capabilityMeta;

const scenarioMeta: Array<{
  id: ModelUseScenario;
  label: string;
  capability: ModelCapability;
}> = [
  { id: 'agent_chat', label: 'Agent 对话', capability: 'chat' },
  { id: 'collection_completion', label: '知识库问答', capability: 'chat' },
  { id: 'collection_embedding', label: '文档向量', capability: 'embedding' },
  { id: 'background_task', label: '后台任务', capability: 'chat' },
];

const scenariosForCapability = (capability: ModelCapability) =>
  scenarioMeta.filter((scenario) => scenario.capability === capability);

const modelCapabilityOptions: ModelCapability[] = [
  'chat',
  'embedding',
];

const modelPresets: ModelPreset[] = [
  {
    providerType: 'openai',
    providerModelId: 'gpt-4.1',
    displayName: 'GPT-4.1',
    capability: 'chat',
    contextWindow: 128000,
    supportsVision: true,
    supportsToolCalling: true,
  },
  {
    providerType: 'openai',
    providerModelId: 'gpt-4.1-mini',
    displayName: 'GPT-4.1 Mini',
    capability: 'chat',
    contextWindow: 128000,
    supportsVision: true,
    supportsToolCalling: true,
  },
  {
    providerType: 'openai',
    providerModelId: 'text-embedding-3-large',
    displayName: 'text-embedding-3-large',
    capability: 'embedding',
    embeddingDimensions: 3072,
  },
  {
    providerType: 'openai',
    providerModelId: 'text-embedding-3-small',
    displayName: 'text-embedding-3-small',
    capability: 'embedding',
    embeddingDimensions: 1536,
  },
  {
    providerType: 'dashscope',
    providerModelId: 'qwen-plus',
    displayName: 'Qwen Plus',
    capability: 'chat',
    contextWindow: 131072,
    supportsToolCalling: true,
  },
  {
    providerType: 'dashscope',
    providerModelId: 'qwen-max',
    displayName: 'Qwen Max',
    capability: 'chat',
    contextWindow: 32768,
    supportsToolCalling: true,
  },
  {
    providerType: 'dashscope',
    providerModelId: 'text-embedding-v4',
    displayName: 'Text Embedding v4',
    capability: 'embedding',
    embeddingDimensions: 1024,
  },
  {
    providerType: 'jina',
    providerModelId: 'jina-embeddings-v3',
    displayName: 'Jina Embeddings v3',
    capability: 'embedding',
    embeddingDimensions: 1024,
  },
  {
    providerType: 'openai_compatible',
    providerModelId: 'google/gemini-2.5-flash',
    displayName: 'Gemini 2.5 Flash',
    capability: 'chat',
    contextWindow: 1048576,
    supportsVision: true,
    supportsToolCalling: true,
  },
  {
    providerType: 'openai_compatible',
    providerModelId: 'qwen/qwen3-32b',
    displayName: 'Qwen3 32B',
    capability: 'chat',
    contextWindow: 131072,
    supportsToolCalling: true,
  },
];

const emptyModelForm: ModelForm = {
  providerModelId: '',
  displayName: '',
  capability: 'chat',
};

const normalize = (value: string) =>
  value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');

const providerFormFromAccount = (
  provider: ModelProvider | undefined,
  account: ModelAccount | undefined,
): ProviderForm => ({
  name: account?.name ?? normalize(provider?.provider_type ?? ''),
  displayName: account?.display_name ?? provider?.display_name ?? '',
  baseUrl: account?.base_url ?? provider?.default_base_url ?? '',
  apiKey: '',
});

const modelKey = (model: Pick<Model, 'provider_model_id' | 'capability'>) =>
  `${model.provider_model_id}:${model.capability}`;

const presetKey = (
  preset: Pick<ModelPreset, 'providerModelId' | 'capability'>,
) => `${preset.providerModelId}:${preset.capability}`;

export function ModelPlatformPanel({ data }: { data: ModelPlatformViewModel }) {
  const router = useRouter();
  const activeProviders = useMemo(
    () => data.providers.filter((provider) => provider.enabled !== false),
    [data.providers],
  );
  const [selectedProviderType, setSelectedProviderType] = useState(
    activeProviders[0]?.provider_type ?? '',
  );
  const selectedProvider =
    activeProviders.find(
      (provider) => provider.provider_type === selectedProviderType,
    ) ?? activeProviders[0];
  const providerAccounts = useMemo(
    () =>
      data.accounts.filter(
        (account) => account.provider_type === selectedProvider?.provider_type,
      ),
    [data.accounts, selectedProvider?.provider_type],
  );
  const primaryAccount = providerAccounts[0];
  const providerModels = useMemo(
    () =>
      data.models.filter((model) =>
        providerAccounts.some((account) => account.id === model.account_id),
      ),
    [data.models, providerAccounts],
  );
  const configuredModelKeys = useMemo(
    () => new Set(providerModels.map(modelKey)),
    [providerModels],
  );
  const providerPresets = useMemo(
    () =>
      modelPresets.filter(
        (preset) => preset.providerType === selectedProvider?.provider_type,
      ),
    [selectedProvider?.provider_type],
  );
  const [providerForm, setProviderForm] = useState<ProviderForm>(
    providerFormFromAccount(selectedProvider, primaryAccount),
  );
  const [modelForm, setModelForm] = useState<ModelForm>(emptyModelForm);
  const [saving, setSaving] = useState<string | null>(null);

  const chooseProvider = (provider: ModelProvider) => {
    const nextAccount = data.accounts.find(
      (account) => account.provider_type === provider.provider_type,
    );
    setSelectedProviderType(provider.provider_type);
    setProviderForm(providerFormFromAccount(provider, nextAccount));
    setModelForm(emptyModelForm);
  };

  const saveProvider = async () => {
    if (!selectedProvider) return;
    const displayName = providerForm.displayName.trim();
    const name = normalize(providerForm.name || selectedProvider.provider_type);

    if (!name || !displayName || !providerForm.baseUrl.trim()) {
      toast.error('请填写显示名称和 Base URL');
      return;
    }
    if (!primaryAccount && !providerForm.apiKey.trim()) {
      toast.error('请填写 API Key');
      return;
    }

    setSaving('provider');
    try {
      if (primaryAccount?.id) {
        await updateModelAccount(primaryAccount.id, {
          name,
          display_name: displayName,
          base_url: providerForm.baseUrl.trim(),
          api_key: providerForm.apiKey.trim() || undefined,
        });
        toast.success('Provider 配置已保存');
      } else {
        await createModelAccount({
          provider_type: selectedProvider.provider_type,
          name,
          display_name: displayName,
          base_url: providerForm.baseUrl.trim(),
          api_key: providerForm.apiKey.trim(),
        });
        toast.success('Provider 已启用');
      }
      setProviderForm({ ...providerForm, apiKey: '' });
      router.refresh();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '保存失败');
    } finally {
      setSaving(null);
    }
  };

  const validateProvider = async () => {
    if (!primaryAccount?.id) return;
    setSaving('validate');
    try {
      const result = await validateModelAccount(primaryAccount.id);
      const message =
        result?.message || (result?.ok ? '连接可用' : '连接不可用');
      if (result?.ok) {
        toast.success(message);
      } else {
        toast.error(message);
      }
      router.refresh();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '校验失败');
    } finally {
      setSaving(null);
    }
  };

  const disableProvider = async () => {
    if (!primaryAccount?.id) return;
    setSaving('disable-provider');
    try {
      await deleteModelAccount(primaryAccount.id);
      toast.success('Provider 已停用');
      router.refresh();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '停用失败');
    } finally {
      setSaving(null);
    }
  };

  const addModel = async (source: ModelPreset | ModelForm) => {
    if (!primaryAccount?.id) {
      toast.error('请先保存 Provider 的 API Key 和 Base URL');
      return;
    }
    const providerModelId = source.providerModelId.trim();
    const displayName = source.displayName.trim();
    const capability = source.capability;

    if (!providerModelId || !displayName) {
      toast.error('请填写模型 ID 和显示名称');
      return;
    }
    if (configuredModelKeys.has(`${providerModelId}:${capability}`)) {
      toast.info('这个模型已经启用过了');
      return;
    }

    setSaving(`model:${providerModelId}:${capability}`);
    try {
      const preset = 'contextWindow' in source ? source : undefined;
      await createModel({
        account_id: primaryAccount.id,
        provider_model_id: providerModelId,
        display_name: displayName,
        capability,
        context_window: preset?.contextWindow,
        embedding_dimensions: preset?.embeddingDimensions,
        supports_vision: Boolean(preset?.supportsVision),
        supports_tool_calling: Boolean(preset?.supportsToolCalling),
        supports_multimodal_embedding: Boolean(
          preset?.supportsMultimodalEmbedding,
        ),
      });
      toast.success('模型已启用');
      if (!preset) setModelForm(emptyModelForm);
      router.refresh();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '启用模型失败');
    } finally {
      setSaving(null);
    }
  };

  const saveUse = async (
    scenario: ModelUseScenario,
    capability: ModelCapability,
    modelId: string,
  ) => {
    setSaving(`use:${scenario}`);
    try {
      await updateModelUse(scenario, {
        capability,
        primary_model_id: modelId,
      });
      toast.success('默认模型已更新');
      router.refresh();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '更新失败');
    } finally {
      setSaving(null);
    }
  };

  const saveAllowedScenario = async (
    model: Model,
    scenario: ModelUseScenario,
    checked: boolean,
  ) => {
    if (!model.id) return;
    const current = new Set(model.allowed_scenarios ?? []);
    if (checked) {
      current.add(scenario);
    } else {
      current.delete(scenario);
    }

    setSaving(`scenario:${model.id}:${scenario}`);
    try {
      await updateModel(model.id, {
        allowed_scenarios: Array.from(current),
      });
      toast.success('可用场景已更新');
      router.refresh();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '更新失败');
    } finally {
      setSaving(null);
    }
  };

  if (!selectedProvider) {
    return (
      <div className="rounded-lg border bg-white p-8">
        <h2 className="text-xl font-semibold">没有可用 Provider</h2>
        <p className="text-muted-foreground mt-2 text-sm">
          系统还没有下发内置 Provider 模板。
        </p>
      </div>
    );
  }

  return (
    <div className="grid gap-5">
      <div className="grid gap-5 lg:grid-cols-[300px_minmax(0,1fr)]">
        <ProviderSidebar
          providers={activeProviders}
          accounts={data.accounts}
          selectedProvider={selectedProvider}
          onSelect={chooseProvider}
        />

        <section className="min-w-0 rounded-lg border bg-white">
          <ProviderHero provider={selectedProvider} account={primaryAccount} />

          <div className="grid gap-8 border-t px-6 py-6">
            <ProviderSettings
              provider={selectedProvider}
              account={primaryAccount}
              form={providerForm}
              saving={saving}
              onChange={setProviderForm}
              onSave={saveProvider}
              onValidate={validateProvider}
              onDisable={disableProvider}
            />

            <ModelFolder
              account={primaryAccount}
              configuredModels={providerModels}
              presets={providerPresets}
              configuredModelKeys={configuredModelKeys}
              modelForm={modelForm}
              saving={saving}
              onModelFormChange={setModelForm}
              onAddModel={addModel}
              onAllowedScenarioChange={saveAllowedScenario}
            />
          </div>
        </section>
      </div>

      <DefaultModels
        models={data.models}
        uses={data.uses}
        saving={saving}
        onChange={saveUse}
      />
    </div>
  );
}

function ProviderSidebar({
  providers,
  accounts,
  selectedProvider,
  onSelect,
}: {
  providers: ModelProvider[];
  accounts: ModelAccount[];
  selectedProvider: ModelProvider;
  onSelect: (provider: ModelProvider) => void;
}) {
  return (
    <aside className="rounded-lg border bg-white">
      <div className="border-b px-5 py-4">
        <div className="text-sm font-semibold">Providers</div>
      </div>
      <div className="grid gap-1 p-2">
        {providers.map((provider) => {
          const providerAccounts = accounts.filter(
            (account) => account.provider_type === provider.provider_type,
          );
          const selected =
            provider.provider_type === selectedProvider.provider_type;

          return (
            <button
              key={provider.provider_type}
              type="button"
              onClick={() => onSelect(provider)}
              className={cn(
                'group grid grid-cols-[minmax(0,1fr)] items-center gap-3 rounded-md border px-4 py-3 text-left transition-colors',
                selected
                  ? 'border-slate-200 bg-slate-100 text-slate-950 shadow-sm'
                  : 'border-transparent hover:bg-slate-50',
              )}
            >
              <span className="min-w-0">
                <span className="block truncate text-sm font-medium">
                  {provider.display_name}
                </span>
                <span
                  className={cn(
                    'text-xs',
                    selected ? 'text-slate-500' : 'text-muted-foreground',
                  )}
                >
                  {providerAccounts.length ? '已启用' : '未启用'}
                </span>
              </span>
            </button>
          );
        })}
      </div>
    </aside>
  );
}

function ProviderHero({
  provider,
  account,
}: {
  provider: ModelProvider;
  account: ModelAccount | undefined;
}) {
  return (
    <div className="px-6 py-6">
      <div className="flex gap-4">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-3xl font-semibold tracking-tight">
              {provider.display_name}
            </h2>
            {account ? (
              <Badge className="bg-emerald-600 hover:bg-emerald-600">
                已启用
              </Badge>
            ) : (
              <Badge variant="secondary">未启用</Badge>
            )}
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            {provider.supported_capabilities.map((capability) => (
              isSupportedModelCapability(capability) ? (
                <CapabilityBadge key={capability} capability={capability} />
              ) : null
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function ProviderSettings({
  provider,
  account,
  form,
  saving,
  onChange,
  onSave,
  onValidate,
  onDisable,
}: {
  provider: ModelProvider;
  account: ModelAccount | undefined;
  form: ProviderForm;
  saving: string | null;
  onChange: (value: ProviderForm) => void;
  onSave: () => void;
  onValidate: () => void;
  onDisable: () => void;
}) {
  return (
    <section className="grid gap-4">
      <SectionTitle title="API Keys" />
      <div className="grid gap-4">
        <div className="grid gap-3 md:grid-cols-2">
          <Field
            label="Display Name"
            value={form.displayName}
            placeholder={provider.display_name}
            onChange={(value) => onChange({ ...form, displayName: value })}
          />
          <Field
            label="Base URL"
            value={form.baseUrl}
            placeholder={provider.default_base_url || 'https://.../v1'}
            onChange={(value) => onChange({ ...form, baseUrl: value })}
          />
        </div>
        <Field
          label="API Key"
          value={form.apiKey}
          placeholder="sk-..."
          type="password"
          onChange={(value) => onChange({ ...form, apiKey: value })}
        />
        <div className="flex flex-wrap items-center justify-between gap-3">
          {account?.validation_error ? (
            <div className="text-muted-foreground text-xs">
              上次校验失败：{account.validation_error}
            </div>
          ) : (
            <div />
          )}
          <div className="ml-auto flex gap-2">
            {account ? (
              <>
                <Button
                  variant="outline"
                  onClick={onValidate}
                  disabled={saving !== null}
                >
                  {saving === 'validate' ? (
                    <Loader2 className="size-4 animate-spin" />
                  ) : null}
                  校验
                </Button>
                <Button
                  variant="outline"
                  onClick={onDisable}
                  disabled={saving !== null}
                >
                  {saving === 'disable-provider' ? (
                    <Loader2 className="size-4 animate-spin" />
                  ) : null}
                  停用
                </Button>
              </>
            ) : null}
            <Button onClick={onSave} disabled={saving !== null}>
              {saving === 'provider' ? (
                <Loader2 className="size-4 animate-spin" />
              ) : null}
              {account ? '保存' : '启用'}
            </Button>
          </div>
        </div>
      </div>
    </section>
  );
}

function ModelFolder({
  account,
  configuredModels,
  presets,
  configuredModelKeys,
  modelForm,
  saving,
  onModelFormChange,
  onAddModel,
  onAllowedScenarioChange,
}: {
  account: ModelAccount | undefined;
  configuredModels: Model[];
  presets: ModelPreset[];
  configuredModelKeys: Set<string>;
  modelForm: ModelForm;
  saving: string | null;
  onModelFormChange: (value: ModelForm) => void;
  onAddModel: (source: ModelPreset | ModelForm) => void;
  onAllowedScenarioChange: (
    model: Model,
    scenario: ModelUseScenario,
    checked: boolean,
  ) => void;
}) {
  return (
    <section className="grid gap-4">
      <SectionTitle title="Models" />
      <div className="overflow-hidden rounded-lg border">
        <div className="grid grid-cols-[minmax(0,1fr)_120px_minmax(220px,1fr)_112px] border-b bg-slate-50 px-4 py-2 text-xs font-medium text-slate-500">
          <span>模型</span>
          <span>能力</span>
          <span>可用场景</span>
          <span aria-hidden />
        </div>
        {configuredModels.map((model) => (
          <ModelRow
            key={model.id ?? model.provider_model_id}
            model={model}
            saving={saving}
            onAllowedScenarioChange={onAllowedScenarioChange}
          />
        ))}
        {presets
          .filter((preset) => !configuredModelKeys.has(presetKey(preset)))
          .map((preset) => {
            const savingKey = `model:${preset.providerModelId}:${preset.capability}`;
            return (
              <PresetRow
                key={`${preset.providerType}:${preset.providerModelId}:${preset.capability}`}
                preset={preset}
                disabled={!account || saving !== null}
                loading={saving === savingKey}
                onAdd={() => onAddModel(preset)}
              />
            );
          })}
        <div className="grid gap-3 border-t bg-white px-4 py-4 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_160px_auto] md:items-end">
          <Field
            label="自定义模型 ID"
            value={modelForm.providerModelId}
            placeholder="provider/model-id"
            onChange={(value) =>
              onModelFormChange({ ...modelForm, providerModelId: value })
            }
          />
          <Field
            label="显示名称"
            value={modelForm.displayName}
            placeholder="My Model"
            onChange={(value) =>
              onModelFormChange({ ...modelForm, displayName: value })
            }
          />
          <div className="grid gap-2">
            <Label>能力</Label>
            <Select
              value={modelForm.capability}
              onValueChange={(capability: ModelCapability) =>
                onModelFormChange({ ...modelForm, capability })
              }
            >
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {modelCapabilityOptions.map((capability) => (
                  <SelectItem key={capability} value={capability}>
                    {capabilityMeta[capability].label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <Button
            onClick={() => onAddModel(modelForm)}
            disabled={!account || saving !== null}
          >
            启用
          </Button>
        </div>
      </div>
    </section>
  );
}

function ModelRow({
  model,
  saving,
  onAllowedScenarioChange,
}: {
  model: Model;
  saving: string | null;
  onAllowedScenarioChange: (
    model: Model,
    scenario: ModelUseScenario,
    checked: boolean,
  ) => void;
}) {
  const scenarios = scenariosForCapability(model.capability);
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_120px_minmax(220px,1fr)_112px] items-center border-b px-4 py-3">
      <div className="min-w-0">
        <div className="truncate text-sm font-medium">{model.display_name}</div>
        <div className="text-muted-foreground mt-1 truncate text-xs">
          {model.provider_model_id}
        </div>
      </div>
      <CapabilityBadge capability={model.capability} />
      <div className="flex flex-wrap gap-3">
        {scenarios.map((scenario) => {
          const savingKey = `scenario:${model.id}:${scenario.id}`;
          return (
            <label
              key={scenario.id}
              className="flex items-center gap-2 text-xs text-slate-700"
            >
              <Switch
                checked={Boolean(
                  model.allowed_scenarios?.includes(scenario.id),
                )}
                disabled={saving !== null || !model.id}
                onCheckedChange={(checked) =>
                  onAllowedScenarioChange(model, scenario.id, checked)
                }
              />
              <span>{scenario.label}</span>
              {saving === savingKey ? (
                <Loader2 className="size-3 animate-spin text-slate-400" />
              ) : null}
            </label>
          );
        })}
      </div>
      <div aria-hidden />
    </div>
  );
}

function PresetRow({
  preset,
  disabled,
  loading,
  onAdd,
}: {
  preset: ModelPreset;
  disabled: boolean;
  loading: boolean;
  onAdd: () => void;
}) {
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_120px_minmax(220px,1fr)_112px] items-center border-b px-4 py-3">
      <div className="min-w-0">
        <div className="truncate text-sm font-medium text-slate-700">
          {preset.displayName}
        </div>
        <div className="text-muted-foreground mt-1 truncate text-xs">
          {preset.providerModelId}
        </div>
      </div>
      <CapabilityBadge capability={preset.capability} />
      <div className="text-muted-foreground text-xs">启用后按默认场景开放</div>
      <div className="flex justify-end">
        <Button size="sm" variant="outline" disabled={disabled} onClick={onAdd}>
          {loading ? <Loader2 className="size-4 animate-spin" /> : null}
          启用
        </Button>
      </div>
    </div>
  );
}

function DefaultModels({
  models,
  uses,
  saving,
  onChange,
}: {
  models: Model[];
  uses: ModelPlatformViewModel['uses'];
  saving: string | null;
  onChange: (
    scenario: ModelUseScenario,
    capability: ModelCapability,
    modelId: string,
  ) => void;
}) {
  const t = useTranslations('page_models');

  return (
    <section className="grid gap-4 rounded-lg border bg-white p-6">
      <SectionTitle title={t('default_model.config')} />
      <div className="grid gap-2 md:grid-cols-2">
        {scenarioMeta.map((scenario) => {
          const candidates = models.filter(
            (model): model is Model & { id: string } =>
              Boolean(model.id) &&
              isModelAllowedForScenario(model, scenario.id),
          );
          const value =
            uses.find((item) => item.scenario === scenario.id)
              ?.primary_model_id ?? '';
          const selectedModel = models.find((model) => model.id === value);
          const selectedIsInvalid =
            Boolean(value) && !candidates.some((model) => model.id === value);
          return (
            <div
              key={scenario.id}
              className={cn(
                'grid gap-2 rounded-lg border bg-white p-3',
                selectedIsInvalid && 'border-red-300 bg-red-50/40',
              )}
            >
              <div className="flex items-center justify-between gap-3">
                <span className="text-sm font-medium">{scenario.label}</span>
                <CapabilityBadge capability={scenario.capability} />
              </div>
              <Select
                value={value}
                onValueChange={(modelId) =>
                  onChange(scenario.id, scenario.capability, modelId)
                }
                disabled={saving !== null || candidates.length === 0}
              >
                <SelectTrigger
                  className={cn(
                    'w-full',
                    selectedIsInvalid && 'border-red-300',
                  )}
                >
                  <SelectValue
                    placeholder={
                      candidates.length ? '选择默认模型' : '暂无可用模型'
                    }
                  />
                </SelectTrigger>
                <SelectContent>
                  {selectedIsInvalid && selectedModel?.id ? (
                    <SelectItem value={selectedModel.id} disabled>
                      {selectedModel.display_name}（不允许当前场景）
                    </SelectItem>
                  ) : null}
                  {candidates.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.display_name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {selectedIsInvalid ? (
                <p className="text-xs text-red-600">
                  当前默认模型未开放给这个场景，请选择一个允许的模型后再保存。
                </p>
              ) : null}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function SectionTitle({
  title,
  description,
}: {
  title: string;
  description?: string;
}) {
  return (
    <div>
      <div className="text-sm font-semibold">{title}</div>
      {description ? (
        <p className="text-muted-foreground mt-1 text-xs">{description}</p>
      ) : null}
    </div>
  );
}

function CapabilityBadge({ capability }: { capability: ModelCapability }) {
  const meta = capabilityMeta[capability];
  return (
    <span className="inline-flex w-fit items-center rounded-md border border-slate-200 bg-white px-2 py-0.5 text-xs font-medium text-slate-600">
      {meta.label}
    </span>
  );
}

function Field({
  label,
  value,
  placeholder,
  type = 'text',
  onChange,
}: {
  label: string;
  value: string;
  placeholder?: string;
  type?: string;
  onChange: (value: string) => void;
}) {
  return (
    <div className="grid gap-2">
      <Label>{label}</Label>
      <Input
        type={type}
        value={value}
        placeholder={placeholder}
        onChange={(event) => onChange(event.currentTarget.value)}
      />
    </div>
  );
}
