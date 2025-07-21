export const model = {
  'model.name': 'Models',
  'model.configuration': 'Model Configuration',
  'model.configuration.description': 'Configure LLM providers and models',
  'model.configuration.admin_only':
    '👤 Admin Notice: Only globally shared models with "recommend" tag are visible to regular users. Private user providers/models are not affected by this restriction.',
  'model.prompt_template': 'Prompt Template',
  'model.llm.tips': 'Large language chat model',
  'model.rerank.tips':
    'Using Reranker after getting the results of a vector query (ANN) can more effectively determine the semantic relevance between documents and queries, re-rank the results more finely, and ultimately improve search quality.',

  // Provider related
  'model.provider.title': 'LLM Providers',
  'model.provider.add': 'Add Provider',
  'model.provider.edit': 'Edit Provider',
  'model.provider.delete': 'Delete Provider',
  'model.provider.manage': 'Manage Models',
  'model.provider.id': 'Provider ID',
  'model.provider.name': 'Name',
  'model.provider.label': 'Display Name',
  'model.provider.url': 'URL',
  'model.provider.base_url': 'API Base URL',
  'model.provider.api_key_short': 'API KEY',
  'model.provider.model_count': 'Model Count',
  'model.provider.type': 'Sharing Scope',
  'model.provider.type.public': 'Global',
  'model.provider.type.user': 'Private',
  'model.provider.completion_dialect': 'Completion API Dialect',
  'model.provider.embedding_dialect': 'Embedding API Dialect',
  'model.provider.rerank_dialect': 'Rerank API Dialect',
  'model.provider.allow_custom_base_url': 'Allow Custom Base URL',
  'model.provider.extra_config': 'Extra Configuration (JSON)',
  'model.provider.delete.confirm':
    'Are you sure to delete provider "{name}"? This will also delete all models under this provider.',

  // Provider status related
  'model.provider.status': 'Status',
  'model.provider.status.enabled': 'Enabled',
  'model.provider.status.disabled': 'Disabled',
  'model.provider.enable': 'Enable',
  'model.provider.disable': 'Disable',
  'model.provider.enable.help':
    'Enable to configure API key and use model services',
  'model.provider.disable.help':
    'Disable to remove API key and stop using model services',

  // Model related
  'model.management.title': '{provider} - Model Management',
  'model.list.title': 'Model List',
  'model.search.placeholder': 'Search model name...',
  'model.add': 'Add Model',
  'model.add.title': 'Add New Model',
  'model.edit': 'Edit Model',
  'model.edit.title': 'Edit Model: {model}',
  'model.delete': 'Delete Model',
  'model.delete.confirm': 'Are you sure to delete model "{name}"?',
  'model.back_to_list': 'Back to List',
  'model.api_type': 'API Type',
  'model.model_name': 'Model Name',
  'model.custom_llm_provider': 'API Dialect',
  'model.max_tokens': 'Max Tokens',
  'model.tags': 'Tags',

  // Form placeholders and validation
  'model.provider.name.placeholder': 'e.g: openai',
  'model.provider.name.required': 'Please enter provider ID',
  'model.provider.label.placeholder': 'e.g: OpenAI',
  'model.provider.label.required': 'Please enter display name',
  'model.provider.base_url.placeholder': 'https://api.openai.com/v1',
  'model.provider.base_url.required': 'Please enter API base URL',
  'model.provider.completion_dialect.placeholder': 'openai',
  'model.provider.completion_dialect.required':
    'Please enter completion API dialect',
  'model.provider.embedding_dialect.placeholder': 'openai',
  'model.provider.embedding_dialect.required':
    'Please enter embedding API dialect',
  'model.provider.rerank_dialect.placeholder': 'jina_ai',
  'model.provider.rerank_dialect.required': 'Please enter rerank API dialect',
  'model.provider.api_key.required': 'Please enter API key',

  'model.provider.required': 'Please select provider',
  'model.api_type.required': 'Please select API type',
  'model.model_name.placeholder': 'e.g: gpt-4o-mini',
  'model.model_name.required': 'Please enter model name',
  'model.custom_llm_provider.placeholder': 'Auto-filled',
  'model.custom_llm_provider.required': 'Please enter API Dialect',
  'model.max_tokens.placeholder': '4096',
  'model.tags.placeholder': 'Enter tags',

  // API types
  'model.api_type.completion': 'Completion',
  'model.api_type.embedding': 'Embedding',
  'model.api_type.rerank': 'Rerank',

  // Messages
  'model.provider.create.success': 'Provider created successfully',
  'model.provider.update.success': 'Provider updated successfully',
  'model.provider.delete.success': 'Provider deleted successfully',
  'model.provider.save.failed': 'Failed to save provider',
  'model.provider.delete.failed': 'Failed to delete provider',
  'model.create.success': 'Model created successfully',
  'model.update.success': 'Model updated successfully',
  'model.delete.success': 'Model deleted successfully',
  'model.save.failed': 'Failed to save model',
  'model.delete.failed': 'Failed to delete model',
  'model.configuration.fetch.failed': 'Failed to fetch LLM configuration',
};

export const model_provider = {
  'model.provider': 'Model Provider',
  'model.provider.required': 'Model Provider is required',
  'model.provider.description':
    'Set the service URI and API Key of the model service provider',

  'model.provider.settings': 'Provider Settings',
  'model.provider.enable': 'Enable',
  'model.provider.disable': 'Disable',
  'model.provider.disable.confirm': 'Are you sure to disable {label} provider?',
  'model.provider.uri': 'Provider URI',
  'model.provider.uri.required': 'Provider URI is required',
  'model.provider.api_key': 'Provider API Key',
  'model.provider.api_key.required': 'Provider API Key is required',
  'model.provider.add': 'Add Provider',

  // API Key management related
  'model.provider.api_key.manage': 'Manage API Keys',
  'model.provider.api_key.description':
    'Configure API keys for model service providers to enable model services',
  'model.provider.api_key.settings': 'API Key Settings',
  'model.provider.api_key.help':
    'Optional: Configure API key for this provider to enable model services',
  'model.provider.api_key.placeholder': 'Enter API key',
  'model.provider.api_key.edit.help':
    'API key is currently configured. Leave empty to keep unchanged, enter new key to update',
  'model.provider.api_key.edit.placeholder':
    'Enter new API key (leave empty to keep unchanged)',
  'model.provider.api_key.configured': 'Configured',
  'model.provider.api_key.not_configured': 'Not Configured',
  'model.provider.api_key.update.success': 'API key updated successfully',
  'model.provider.api_key.update.failed': 'Failed to update API key',
  'model.provider.disable.success': 'Provider disabled successfully',
  'model.provider.disable.failed': 'Failed to disable provider',
  'model.provider.fetch.failed': 'Failed to fetch provider information',
};
