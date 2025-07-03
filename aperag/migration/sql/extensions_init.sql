-- PostgreSQL Extensions Initialization
-- This script creates necessary extensions for ApeRAG
-- Extensions must be created before schema tables that use them

-- Create pgvector extension for vector operations
-- Used by LightRAG tables: lightrag_doc_chunks, lightrag_vdb_entity, lightrag_vdb_relation
CREATE EXTENSION IF NOT EXISTS vector;

-- Optional: Create other useful extensions
-- Uncomment as needed based on project requirements

-- CREATE EXTENSION IF NOT EXISTS "uuid-ossp";  -- For UUID generation
-- CREATE EXTENSION IF NOT EXISTS pg_trgm;      -- For trigram text search
-- CREATE EXTENSION IF NOT EXISTS btree_gin;    -- For GIN indexes on btree data
-- CREATE EXTENSION IF NOT EXISTS btree_gist;   -- For GIST indexes on btree data 