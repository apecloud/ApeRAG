export const model = {
  'model.name': '模型',
  'model.configuration': '模型配置',
  'model.configuration.description': '配置LLM提供商和模型',

  'model.prompt_template': '提示模版',
  'model.provider.type': '共享范围',
  'model.provider.type.public': '全局共享',
  'model.provider.type.user': '私有',
  'model.llm.tips': '大型语言对话模型',
  'model.rerank.tips':
    '在拿到向量查询（ANN）的结果后使用 Reranker，能够更有效地确定文档和查询之间的语义相关性，更精细地对结果重排，最终提高搜索质量。',

  // Provider related
  'model.provider.title': 'LLM提供商',
  'model.provider.add': '添加提供商',
  'model.provider.edit': '编辑提供商',
  'model.provider.delete': '删除提供商',
  'model.provider.manage': '管理模型',
  'model.provider.id': '服务商ID',
  'model.provider.name': '名称',
  'model.provider.label': '显示名称',
  'model.provider.url': 'URL',
  'model.provider.base_url': 'API基础URL',
  'model.provider.api_key_short': 'API KEY',
  'model.provider.model_count': '模型数量',
  'model.provider.completion_dialect': '对话API方言',
  'model.provider.embedding_dialect': '嵌入API方言',
  'model.provider.rerank_dialect': '重排API方言',
  'model.provider.allow_custom_base_url': '允许自定义基础URL',
  'model.provider.extra_config': '额外配置 (JSON)',
  'model.provider.delete.confirm':
    '确定要删除提供商 "{name}" 吗？这将同时删除该提供商下的所有模型。',

  // Provider status related
  'model.provider.status': '状态',
  'model.provider.status.enabled': '启用',
  'model.provider.status.disabled': '禁用',
  'model.provider.enable': '启用',
  'model.provider.disable': '禁用',
  'model.provider.enable.help': '启用后可配置API密钥使用模型服务',
  'model.provider.disable.help': '禁用后将删除API密钥，无法使用模型服务',

  // Model related
  'model.management.title': '{provider} - 模型管理',
  'model.list.title': '模型列表',
  'model.search.placeholder': '搜索模型名称...',
  'model.add': '添加模型',
  'model.add.title': '添加新模型',
  'model.edit': '编辑模型',
  'model.edit.title': '编辑模型：{model}',
  'model.delete': '删除模型',
  'model.delete.confirm': '确定要删除模型 "{name}" 吗？',
  'model.back_to_list': '返回列表',
  'model.api_type': 'API类型',
  'model.model_name': '模型名称',
  'model.custom_llm_provider': 'API方言',
  'model.max_tokens': '最大Token数',
  'model.tags': '标签',

  // Form placeholders and validation
  'model.provider.name.placeholder': '例如: openai',
  'model.provider.name.required': '请输入提供商ID',
  'model.provider.label.placeholder': '例如: OpenAI',
  'model.provider.label.required': '请输入显示名称',
  'model.provider.base_url.placeholder': 'https://api.openai.com/v1',
  'model.provider.base_url.required': '请输入API基础URL',
  'model.provider.completion_dialect.placeholder': 'openai',
  'model.provider.completion_dialect.required': '请输入对话API方言',
  'model.provider.embedding_dialect.placeholder': 'openai',
  'model.provider.embedding_dialect.required': '请输入嵌入API方言',
  'model.provider.rerank_dialect.placeholder': 'jina_ai',
  'model.provider.rerank_dialect.required': '请输入重排API方言',
  'model.provider.api_key.required': '请输入API密钥',

  'model.provider.required': '请选择提供商',
  'model.api_type.required': '请选择API类型',
  'model.model_name.placeholder': '例如: gpt-4o-mini',
  'model.model_name.required': '请输入模型名称',
  'model.custom_llm_provider.placeholder': '自动填充',
  'model.custom_llm_provider.required': '请输入API方言',
  'model.max_tokens.placeholder': '4096',
  'model.tags.placeholder': '输入标签',

  // API types
  'model.api_type.completion': 'Completion',
  'model.api_type.embedding': 'Embedding',
  'model.api_type.rerank': 'Rerank',

  // Messages
  'model.provider.create.success': '提供商创建成功',
  'model.provider.update.success': '提供商更新成功',
  'model.provider.delete.success': '提供商删除成功',
  'model.provider.save.failed': '提供商保存失败',
  'model.provider.delete.failed': '提供商删除失败',
  'model.create.success': '模型创建成功',
  'model.update.success': '模型更新成功',
  'model.delete.success': '模型删除成功',
  'model.save.failed': '模型保存失败',
  'model.delete.failed': '模型删除失败',
  'model.configuration.fetch.failed': '获取LLM配置失败',

  // Use cases
  'model.usecase.collection': '知识库',
  'model.usecase.agent': '智能代理',
};

export const model_provider = {
  'model.provider': '模型服务商',
  'model.provider.required': '请选择模型服务商',
  'model.provider.description': '设置模型服务商的服务URI及API Key',

  'model.provider.settings': '服务商设置',
  'model.provider.enable': '启用',
  'model.provider.disable': '禁用',
  'model.provider.disable.confirm': '确定禁用 {label} 吗？',
  'model.provider.uri': '模型服务商 URI',
  'model.provider.uri.required': '请输入模型服务商URI',
  'model.provider.api_key': '模型服务商API Key',
  'model.provider.api_key.required': '请输入模型服务商API Key',
  'model.provider.add': '添加服务商',

  // API Key管理相关
  'model.provider.api_key.manage': '管理API密钥',
  'model.provider.api_key.description': '配置模型服务商的API密钥以启用模型服务',
  'model.provider.api_key.settings': 'API密钥设置',
  'model.provider.api_key.help': '可选：配置此服务商的API密钥以启用模型服务',
  'model.provider.api_key.placeholder': '输入API密钥',
  'model.provider.api_key.edit.help':
    '当前已配置API密钥。留空保持不变，输入新密钥以更新',
  'model.provider.api_key.edit.placeholder': '输入新的API密钥（留空保持不变）',
  'model.provider.api_key.configured': '已配置',
  'model.provider.api_key.not_configured': '未配置',
  'model.provider.api_key.update.success': 'API密钥更新成功',
  'model.provider.api_key.update.failed': 'API密钥更新失败',
  'model.provider.disable.success': '服务商禁用成功',
  'model.provider.disable.failed': '服务商禁用失败',
  'model.provider.fetch.failed': '获取服务商信息失败',
};

export const model_configuration = {
  'model.configuration': '模型配置',
  'model.configuration.description': '配置LLM提供商和模型',

  // LLM Provider related
  'llm.provider': 'LLM提供商',
  'llm.provider.name': '提供商名称',
  'llm.provider.name.placeholder': '例如: openai',
  'llm.provider.label': '显示名称',
  'llm.provider.label.placeholder': '例如: OpenAI',
  'llm.provider.base_url': 'API基础URL',
  'llm.provider.base_url.placeholder': 'https://api.openai.com/v1',
  'llm.provider.allow_custom_base_url': '允许自定义URL',
  'llm.provider.completion_dialect': '对话API方言',
  'llm.provider.embedding_dialect': '嵌入API方言',
  'llm.provider.rerank_dialect': '重排API方言',
  'llm.provider.extra': '额外配置 (JSON)',
  'llm.provider.extra.placeholder': '{"key": "value"}',
  'llm.provider.add': '添加提供商',
  'llm.provider.edit': '编辑提供商',
  'llm.provider.delete.confirm':
    '确定要删除提供商 "{label}" 吗？这将同时删除该提供商下的所有模型。',

  // LLM Model related
  'llm.model': 'LLM模型',
  'llm.model.provider': '提供商',
  'llm.model.api_type': 'API类型',
  'llm.model.model_name': '模型名称',
  'llm.model.model_name.placeholder': '例如: gpt-4o-mini',
  'llm.model.custom_llm_provider': 'API方言',
  'llm.model.custom_llm_provider.placeholder': '自动填充',
  'llm.model.max_tokens': '最大Token数',
  'llm.model.max_tokens.placeholder': '4096',
  'llm.model.tags': '标签',
  'llm.model.tags.placeholder': '输入标签',
  'llm.model.add': '添加模型',
  'llm.model.edit': '编辑模型',
  'llm.model.delete.confirm': '确定要删除模型 "{model}" 吗？',
  'llm.model.add.for.provider': '添加模型',

  // API types
  'llm.api.completion': 'Completion',
  'llm.api.embedding': 'Embedding',
  'llm.api.rerank': 'Rerank',

  // Form validation messages
  'llm.provider.name.required': '请输入提供商名称',
  'llm.provider.label.required': '请输入显示名称',
  'llm.provider.base_url.required': '请输入API基础URL',
  'llm.model.provider.required': '请选择提供商',
  'llm.model.api_type.required': '请选择API类型',
  'llm.model.model_name.required': '请输入模型名称',
  'llm.model.custom_llm_provider.required': '请输入API方言',

  // Success messages
  'llm.provider.created': '提供商创建成功',
  'llm.provider.updated': '提供商更新成功',
  'llm.provider.deleted': '提供商删除成功',
  'llm.model.created': '模型创建成功',
  'llm.model.updated': '模型更新成功',
  'llm.model.deleted': '模型删除成功',

  // Error messages
  'llm.configuration.fetch.error': '获取LLM配置失败',
  'llm.provider.save.error': '保存提供商失败',
  'llm.provider.delete.error': '删除提供商失败',
  'llm.model.save.error': '保存模型失败',
  'llm.model.delete.error': '删除模型失败',
};
