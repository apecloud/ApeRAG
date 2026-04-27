#!/bin/bash
# fix_local_providers.sh
# Run this after adding any new local Ollama provider in ApeRAG UI
# Usage: ./fix_local_providers.sh

set -e

POSTGRES_CONTAINER="aperag-postgres"
POSTGRES_USER="postgres"
POSTGRES_DB="postgres"

psql() {
  docker exec -i $POSTGRES_CONTAINER \
    psql -U $POSTGRES_USER -d $POSTGRES_DB "$@"
}

echo "=== Fixing local Ollama providers in ApeRAG ==="

# 1. Fix custom_llm_provider for all local providers
echo "→ Setting custom_llm_provider = 'openai' for all local providers..."
psql -c "
  UPDATE llm_provider_models 
  SET custom_llm_provider = 'openai'
  WHERE provider_name IN (
    SELECT name FROM llm_provider 
    WHERE base_url LIKE '%172.17.0.1%'
       OR base_url LIKE '%localhost%'
       OR base_url LIKE '%127.0.0.1%'
  );
"

# 2. Make local providers public
echo "→ Making local providers public..."
psql -c "
  UPDATE llm_provider 
  SET user_id = 'public'
  WHERE base_url LIKE '%172.17.0.1%'
     OR base_url LIKE '%localhost%'
     OR base_url LIKE '%127.0.0.1%';
"

# 3. Set default tags for completion models
echo "→ Setting default_for_background_task tag on first local completion model..."
psql -c "
  UPDATE llm_provider_models
  SET tags = tags::jsonb || '[\"default_for_background_task\"]'::jsonb
  WHERE provider_name IN (
    SELECT name FROM llm_provider 
    WHERE base_url LIKE '%172.17.0.1%'
       OR base_url LIKE '%localhost%'
  )
  AND api = 'completion'
  AND NOT (tags::jsonb @> '[\"default_for_background_task\"]'::jsonb);
"

# 4. Set default tags for embedding models
echo "→ Setting default_for_embedding tag on local embedding models..."
psql -c "
  UPDATE llm_provider_models
  SET tags = tags::jsonb || '[\"default_for_embedding\"]'::jsonb
  WHERE provider_name IN (
    SELECT name FROM llm_provider 
    WHERE base_url LIKE '%172.17.0.1%'
       OR base_url LIKE '%localhost%'
  )
  AND api = 'embedding'
  AND NOT (tags::jsonb @> '[\"default_for_embedding\"]'::jsonb);
"

# 5. Show current state
echo ""
echo "=== Current local provider models ==="
psql -c "
  SELECT 
    lp.name as provider,
    lp.base_url,
    lpm.model,
    lpm.api,
    lpm.custom_llm_provider,
    lpm.tags
  FROM llm_provider_models lpm
  JOIN llm_provider lp ON lp.name = lpm.provider_name
  WHERE lp.base_url LIKE '%172.17.0.1%'
     OR lp.base_url LIKE '%localhost%'
  ORDER BY lp.name, lpm.api, lpm.model;
"

echo ""
echo "=== Done. Restart ApeRAG services: ==="
echo "  cd ~/ApeRAG && docker compose restart api celeryworker celerybeat"
