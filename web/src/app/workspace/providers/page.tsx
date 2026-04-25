import {
  PageContainer,
  PageContent,
  PageHeader,
} from '@/components/page-container';
import { getModelPlatform } from '@/features/providers/server-api';
import type { Metadata } from 'next';
import { getTranslations } from 'next-intl/server';
import { ModelPlatformPanel } from './model-platform-panel';

export async function generateMetadata(): Promise<Metadata> {
  return {
    title: '大模型配置',
  };
}

export default async function Page() {
  const page_models = await getTranslations('page_models');
  const data = await getModelPlatform();

  return (
    <PageContainer>
      <PageHeader
        breadcrumbs={[{ title: page_models('metadata.provider_title') }]}
      />
      <PageContent className="max-w-7xl px-5 py-8 md:px-8 md:py-10">
        <div className="mb-8">
          <div className="text-muted-foreground font-mono text-[11px] tracking-[0.12em] uppercase">
            Model Platform
          </div>
          <h1 className="font-serif mt-2 text-4xl leading-none font-normal tracking-normal md:text-[44px]">
            大模型配置
          </h1>
          <p className="text-muted-foreground mt-3 max-w-2xl text-sm leading-6">
            配置模型服务商、模型账号、模型和模型用途。这里不再要求填写 LiteLLM dialect 或内部 provider 路由名。
          </p>
        </div>
        <ModelPlatformPanel data={data} />
      </PageContent>
    </PageContainer>
  );
}
