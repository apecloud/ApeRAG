import { createServerApiClient } from '@/lib/api/typed/server';
import type {
  Model,
  ModelAccount,
  ModelPlatformViewModel,
  ModelProvider,
  ModelUse,
} from './types';

export async function getModelPlatform(): Promise<ModelPlatformViewModel> {
  const client = (await createServerApiClient()) as any;
  const [providers, accounts, models, uses] = await Promise.all([
    client.GET('/api/v2/model-providers'),
    client.GET('/api/v2/model-accounts'),
    client.GET('/api/v2/models'),
    client.GET('/api/v2/model-uses'),
  ]);

  return {
    providers: (providers.data?.items ?? []) as ModelProvider[],
    accounts: (accounts.data?.items ?? []) as ModelAccount[],
    models: (models.data?.items ?? []) as Model[],
    uses: (uses.data?.items ?? []) as ModelUse[],
  };
}
