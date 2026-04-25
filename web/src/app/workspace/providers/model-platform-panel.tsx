'use client';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  createModel,
  createModelAccount,
  updateModelUse,
} from '@/features/providers/client-api';
import type {
  Model,
  ModelAccount,
  ModelCapability,
  ModelPlatformViewModel,
  ModelUseScenario,
} from '@/features/providers/types';
import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { toast } from 'sonner';

const scenarios: { id: ModelUseScenario; label: string; capability: ModelCapability }[] = [
  { id: 'agent_chat', label: 'Agent 对话模型', capability: 'chat' },
  { id: 'collection_completion', label: '知识库补全模型', capability: 'chat' },
  { id: 'collection_embedding', label: '知识库向量模型', capability: 'embedding' },
  { id: 'retrieval_rerank', label: '检索重排模型', capability: 'rerank' },
  { id: 'background_task', label: '后台任务模型', capability: 'chat' },
];

export function ModelPlatformPanel({ data }: { data: ModelPlatformViewModel }) {
  const router = useRouter();
  const [account, setAccount] = useState({
    provider_type: data.providers[0]?.provider_type ?? 'openai_compatible',
    name: '',
    display_name: '',
    base_url: data.providers[0]?.default_base_url ?? '',
    api_key: '',
  });
  const [model, setModel] = useState({
    account_id: data.accounts[0]?.id ?? '',
    provider_model_id: '',
    display_name: '',
    capability: 'chat' as ModelCapability,
  });

  async function saveAccount() {
    await createModelAccount(account);
    toast.success('模型账号已创建');
    router.refresh();
  }

  async function saveModel() {
    await createModel(model);
    toast.success('模型已添加');
    router.refresh();
  }

  async function saveUse(scenario: ModelUseScenario, capability: ModelCapability, modelId: string) {
    await updateModelUse(scenario, { capability, primary_model_id: modelId });
    toast.success('模型用途已更新');
    router.refresh();
  }

  return (
    <div className="grid gap-6">
      <section className="grid gap-4 md:grid-cols-3">
        <Metric title="模型服务商" value={data.providers.length} />
        <Metric title="模型账号" value={data.accounts.length} />
        <Metric title="模型" value={data.models.length} />
      </section>

      <Card>
        <CardHeader>
          <CardTitle>添加模型账号</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-3 md:grid-cols-5">
          <Select
            value={account.provider_type}
            onValueChange={(providerType) => {
              const provider = data.providers.find((item) => item.provider_type === providerType);
              setAccount((value) => ({
                ...value,
                provider_type: providerType,
                base_url: provider?.default_base_url ?? value.base_url,
              }));
            }}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {data.providers.map((provider) => (
                <SelectItem key={provider.provider_type} value={provider.provider_type}>
                  {provider.display_name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Input placeholder="账号标识" value={account.name} onChange={(e) => setAccount({ ...account, name: e.target.value })} />
          <Input
            placeholder="显示名称"
            value={account.display_name}
            onChange={(e) => setAccount({ ...account, display_name: e.target.value })}
          />
          <Input
            placeholder="Base URL"
            value={account.base_url}
            onChange={(e) => setAccount({ ...account, base_url: e.target.value })}
          />
          <Input
            placeholder="API Key"
            type="password"
            value={account.api_key}
            onChange={(e) => setAccount({ ...account, api_key: e.target.value })}
          />
          <Button className="md:col-span-5" onClick={saveAccount}>
            保存模型账号
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>添加模型</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-3 md:grid-cols-5">
          <Select value={model.account_id} onValueChange={(accountId) => setModel({ ...model, account_id: accountId })}>
            <SelectTrigger>
              <SelectValue placeholder="选择模型账号" />
            </SelectTrigger>
            <SelectContent>
              {data.accounts.map((item) => (
                <SelectItem key={item.id ?? item.name} value={item.id ?? ''}>
                  {item.display_name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Input
            placeholder="Provider Model ID"
            value={model.provider_model_id}
            onChange={(e) => setModel({ ...model, provider_model_id: e.target.value })}
          />
          <Input
            placeholder="显示名称"
            value={model.display_name}
            onChange={(e) => setModel({ ...model, display_name: e.target.value })}
          />
          <Select value={model.capability} onValueChange={(capability: ModelCapability) => setModel({ ...model, capability })}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="chat">对话</SelectItem>
              <SelectItem value="embedding">向量</SelectItem>
              <SelectItem value="rerank">重排</SelectItem>
            </SelectContent>
          </Select>
          <Button onClick={saveModel}>添加模型</Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>模型用途</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-3">
          {scenarios.map((scenario) => (
            <ModelUseRow
              key={scenario.id}
              label={scenario.label}
              capability={scenario.capability}
              models={data.models}
              value={data.uses.find((item) => item.scenario === scenario.id)?.primary_model_id ?? ''}
              onChange={(modelId) => saveUse(scenario.id, scenario.capability, modelId)}
            />
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>已配置模型</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-2">
          {data.models.map((item) => (
            <div key={item.id} className="flex items-center justify-between rounded-lg border p-3">
              <div>
                <div className="font-medium">{item.display_name}</div>
                <div className="text-muted-foreground text-xs">{item.provider_model_id}</div>
              </div>
              <div className="text-muted-foreground text-xs">{item.capability}</div>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}

function Metric({ title, value }: { title: string; value: number }) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="text-muted-foreground text-xs">{title}</div>
        <div className="mt-2 text-2xl font-medium">{value}</div>
      </CardContent>
    </Card>
  );
}

function ModelUseRow({
  label,
  capability,
  models,
  value,
  onChange,
}: {
  label: string;
  capability: ModelCapability;
  models: Model[];
  value: string;
  onChange: (modelId: string) => void;
}) {
  const candidates = models.filter((model) => model.capability === capability);
  return (
    <div className="grid items-center gap-3 md:grid-cols-[220px_1fr]">
      <div className="text-sm font-medium">{label}</div>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger>
          <SelectValue placeholder="选择模型" />
        </SelectTrigger>
        <SelectContent>
          {candidates.map((model) => (
            <SelectItem key={model.id ?? model.provider_model_id} value={model.id ?? ''}>
              {model.display_name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
