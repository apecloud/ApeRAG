import {
  BotTypeEnum,
  NodeData,
  WorkflowDefinition,
  WorkflowStyle,
} from '@/api';
import { ApeEdge, ApeNode, ApeNodeHandlePosition } from '@/types';
import Dagre from '@dagrejs/dagre';
import { Position } from '@xyflow/react';
import uniqid from 'uniqid';

const getNodeHandlePositions = (
  direction: WorkflowStyle['layoutDirection'] | undefined = 'LR',
): ApeNodeHandlePosition => {
  const positions: ApeNodeHandlePosition = {};
  switch (direction) {
    case 'TB':
      Object.assign(positions, {
        sourcePosition: Position.Bottom,
        targetPosition: Position.Top,
      });
      break;
    case 'LR':
      Object.assign(positions, {
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
      });
      break;
  }
  return positions;
};

export const getLayoutedElements = (
  nodes: ApeNode[],
  edges: ApeEdge[],
  options: { direction: WorkflowStyle['layoutDirection'] },
): {
  nodes: ApeNode[];
  edges: ApeEdge[];
} => {
  const direction = options.direction;
  const g = new Dagre.graphlib.Graph().setDefaultEdgeLabel(() => ({}));
  g.setGraph({
    rankdir: direction,
    nodesep: 60,
    ranksep: 100,
  });
  edges.forEach((edge) => g.setEdge(edge.source, edge.target));
  nodes.forEach((node) => {
    g.setNode(node.id, {
      ...node,
      width: node.measured?.width ?? 0,
      height: node.measured?.height ?? 0,
    });
  });
  Dagre.layout(g);
  return {
    nodes: nodes.map((node) => {
      const position = g.node(node.id);
      // We are shifting the dagre node position (anchor=center center) to the top left
      // so it matches the React Flow node anchor point (top left).
      const x = position.x - (node.measured?.width ?? 0) / 2;
      const y = position.y - (node.measured?.height ?? 0) / 2;
      return {
        ...node,
        position: { x, y },
        ...getNodeHandlePositions(direction),
      };
    }),
    edges,
  };
};

// start schema
export const nodeStartDefinition = (): NodeData => ({
  input: {
    schema: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: "User's question or query",
        },
      },
      required: ['query'],
    },
    values: {
      query: '',
    },
  },
  output: {
    schema: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: "User's question or query",
        },
      },
      required: ['query'],
    },
  },
});

// vector_search schema
export const nodeVectorSearchDefinition = (params?: {
  startId?: string;
}): NodeData => ({
  input: {
    schema: {
      type: 'object',
      properties: {
        top_k: {
          type: 'integer',
          default: 5,
          minimum: 1,
          maximum: 10,
          description: 'Number of top results to return',
        },
        similarity_threshold: {
          type: 'number',
          default: 0.7,
          minimum: 0.1,
          maximum: 1,
          description: 'Similarity threshold for vector search',
        },
        query: {
          type: 'string',
          description: "User's question or query",
        },
        collection_ids: {
          type: 'array',
          items: {
            type: 'string',
          },
        },
      },
      required: ['top_k', 'similarity_threshold', 'query', 'collection_ids'],
    },
    values: {
      top_k: 5,
      similarity_threshold: 0.7,
      query: params?.startId
        ? `{{ nodes.${params.startId}.output.query }}`
        : '',
    },
  },
  output: {
    schema: {
      type: 'object',
      properties: {
        docs: {
          type: 'array',
          description: 'Docs from vector search',
          items: {
            $ref: '#/schema/document_with_score',
          },
        },
      },
      required: ['docs'],
    },
  },
});

// fulltext_search schema
export const nodeFulltextSearchDefinition = (params?: {
  startId?: string;
}): NodeData => ({
  input: {
    schema: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: "User's question or query",
        },
        top_k: {
          type: 'integer',
          default: 3,
          minimum: 1,
          maximum: 10,
          description: 'Number of top results to return',
        },
        collection_ids: {
          type: 'array',
          items: {
            type: 'string',
          },
        },
      },
      required: ['query', 'top_k', 'collection_ids'],
    },
    values: {
      query: params?.startId
        ? `{{ nodes.${params.startId}.output.query }}`
        : '',
      top_k: 3,
    },
  },
  output: {
    schema: {
      type: 'object',
      properties: {
        docs: {
          type: 'array',
          description: 'Docs from fulltext search',
          items: {
            $ref: '#/schema/document_with_score',
          },
        },
      },
      required: ['docs'],
    },
  },
});

// graph_search node
export const nodeGraphSearchDefinition = (): NodeData => ({
  input: {
    schema: {
      type: 'object',
      properties: {
        top_k: {
          type: 'integer',
          default: 5,
          minimum: 1,
          maximum: 10,
          description: 'Number of top results to return',
        },
        collection_ids: {
          type: 'array',
          description: 'Collection IDs',
          items: {
            type: 'string',
          },
          default: [],
        },
      },
      required: ['top_k', 'collection_ids'],
    },
    values: {
      top_k: 5,
      collection_ids: [],
    },
  },
  output: {
    schema: {
      type: 'object',
      properties: {
        docs: {
          type: 'array',
          description: 'Docs from graph search',
          items: {
            $ref: '#/schema/document_with_score',
          },
        },
      },
      required: ['docs'],
    },
  },
});

// merge schema
export const nodeMergeDefinition = (params?: {
  vectorSearchId: string;
  fulltextSearchId: string;
  graphSearchId?: string;
}): NodeData => ({
  input: {
    schema: {
      type: 'object',
      properties: {
        merge_strategy: {
          type: 'string',
          default: 'union',
          enum: ['union', 'intersection'],
          description: 'How to merge results',
        },
        deduplicate: {
          type: 'boolean',
          default: true,
          description: 'Whether to deduplicate merged results',
        },
        vector_search_docs: {
          type: 'array',
          description: 'Docs from vector search',
          items: {
            $ref: '#/schema/document_with_score',
          },
        },
        fulltext_search_docs: {
          type: 'array',
          description: 'Docs from fulltext search',
          items: {
            $ref: '#/schema/document_with_score',
          },
        },
        graph_search_docs: {
          type: 'array',
          description: 'Docs from graph search',
          items: {
            $ref: '#/schema/document_with_score',
          },
        },
      },
      required: [
        'merge_strategy',
        'deduplicate',
        'vector_search_docs',
        'fulltext_search_docs',
        'graph_search_docs',
      ],
    },
    values: {
      merge_strategy: 'union',
      deduplicate: true,
      vector_search_docs: params?.vectorSearchId
        ? `{{ nodes.${params.vectorSearchId}.output.docs }}`
        : [],
      fulltext_search_docs: params?.fulltextSearchId
        ? `{{ nodes.${params.fulltextSearchId}.output.docs }}`
        : [],
      graph_search_docs: params?.graphSearchId
        ? `{{ nodes.${params.graphSearchId}.output.docs }}`
        : [],
    },
  },
  output: {
    schema: {
      type: 'object',
      properties: {
        docs: {
          type: 'array',
          description: 'Docs after merge',
          items: {
            $ref: '#/schema/document_with_score',
          },
        },
      },
      required: ['docs'],
    },
  },
});

// rerank schema
export const nodeRerankDefinition = (params?: { docId: string }): NodeData => ({
  input: {
    schema: {
      type: 'object',
      properties: {
        model: {
          type: 'string',
          default: 'bge-reranker',
          description: 'Rerank model name',
        },
        model_service_provider: {
          type: 'string',
          default: 'openai',
          description: 'model service provider',
        },
        custom_llm_provider: {
          type: 'string',
          default: 'jina_ai',
          description: 'custom llm provider',
        },
        docs: {
          type: 'array',
          description: 'Docs to rerank',
          items: {
            $ref: '#/schema/document_with_score',
          },
        },
      },
      required: [
        'model',
        'model_service_provider',
        'custom_llm_provider',
        'docs',
      ],
    },
    values: {
      model: '',
      model_service_provider: '',
      custom_llm_provider: '',
      docs: params?.docId ? `{{ nodes.${params.docId}.output.docs }}` : [],
    },
  },
  output: {
    schema: {
      type: 'object',
      properties: {
        docs: {
          type: 'array',
          description: 'Docs after rerank',
          items: {
            $ref: '#/schema/document_with_score',
          },
        },
      },
      required: ['docs'],
    },
  },
});

// llm schema
export const nodeLlmDefinition = (params?: {
  startId?: string;
  docId?: string;
  botType?: BotTypeEnum;
}): NodeData => ({
  input: {
    schema: {
      type: 'object',
      properties: {
        model_service_provider: {
          type: 'string',
          default: 'openrouter',
          description: 'model service provider',
        },
        model_name: {
          type: 'string',
          default: 'deepseek/deepseek-v3-base:free',
          description: 'model name',
        },
        custom_llm_provider: {
          type: 'string',
          default: 'openai',
          description: 'custom llm provider',
        },
        prompt_template: {
          type: 'string',
          default: '{context}\n{query}',
          description: 'Prompt template',
        },
        temperature: {
          type: 'number',
          default: 0.7,
          minimum: 0,
          maximum: 1,
          description: 'Sampling temperature',
        },
        query: {
          type: 'string',
          description: "User's question or query",
        },
        docs: {
          type: 'array',
          description: 'Docs for LLM context',
          items: {
            $ref: '#/schema/document_with_score',
          },
        },
      },
      required: [
        'model_service_provider',
        'model_name',
        'custom_llm_provider',
        'prompt_template',
        'temperature',
        'query',
        'docs',
      ],
    },
    values: {
      model_service_provider: '',
      model_name: '',
      custom_llm_provider: '',
      prompt_template:
        params?.botType === 'knowledge' ? '{context}\n{query}' : '{query}',
      temperature: 0.7,
      query: params?.startId
        ? `{{ nodes.${params.startId}.output.query }}`
        : '',
      docs: params?.docId ? `{{ nodes.${params.docId}.output.docs }}` : [],
    },
  },
  output: {
    schema: {
      type: 'object',
      properties: {
        text: {
          type: 'string',
          description: 'text generated by LLM',
        },
      },
    },
  },
});

// workflow schema
export const workflow_definition: WorkflowDefinition = {
  name: 'rag_flow',
  title: 'RAG Knowledge Base Flow',
  description: 'A typical RAG flow with parallel retrieval and reranking',
  version: '1.0.0',
  execution: {
    timeout: 300,
    retry: {
      max_attempts: 3,
      delay: 5,
    },
    error_handling: {
      strategy: 'stop_on_error',
      notification: {
        email: ['admin@example.com'],
      },
    },
  },
  schema: {
    document_with_score: {
      type: 'object',
      properties: {
        doc_id: {
          type: 'string',
        },
        text: {
          type: 'string',
        },
        score: {
          type: 'number',
        },
        metadata: {
          type: 'object',
        },
      },
    },
  },
  nodes: [],
  edges: [],
  style: {
    edgeType: 'default',
    layoutDirection: 'LR',
  },
};

// workflow schema form common bot
export const getBotCommonWorkflow = (
  type?: BotTypeEnum,
): WorkflowDefinition => {
  const startId = uniqid();
  const llmId = uniqid();
  return {
    ...workflow_definition,
    nodes: [
      {
        id: startId,
        type: 'start',
        data: nodeStartDefinition(),
        position: { x: 0, y: 238 },
        deletable: false,
        dragHandle: '.drag-handle',
      },
      {
        id: llmId,
        type: 'llm',
        data: nodeLlmDefinition({ startId, botType: type }),
        position: { x: 422, y: 0 },
        dragHandle: '.drag-handle',
        deletable: false,
      },
    ],
    edges: [
      {
        id: uniqid(),
        source: startId,
        target: llmId,
        type: 'default',
      },
    ],
  };
};

// workflow schema for knowledge bot
export const getBotKnowledgeWorkflow = (
  type?: BotTypeEnum,
): WorkflowDefinition => {
  const startId = uniqid();
  const vectorSearchId = uniqid();
  const fulltextSearchId = uniqid();
  const graphSearchId = uniqid();
  const mergeId = uniqid();
  const rerankId = uniqid();
  const llmId = uniqid();

  return {
    ...workflow_definition,
    nodes: [
      {
        id: startId,
        type: 'start',
        data: nodeStartDefinition(),
        position: { x: 0, y: 719 },
        deletable: false,
        dragHandle: '.drag-handle',
      },
      {
        id: vectorSearchId,
        data: nodeVectorSearchDefinition({ startId }),
        position: { x: 422, y: 0 },
        type: 'vector_search',
        dragHandle: '.drag-handle',
        deletable: false,
      },
      {
        id: fulltextSearchId,
        type: 'fulltext_search',
        data: nodeFulltextSearchDefinition({ startId }),
        position: { x: 422, y: 610 },
        dragHandle: '.drag-handle',
        deletable: false,
      },
      {
        id: graphSearchId,
        type: 'graph_search',
        data: nodeGraphSearchDefinition(),
        position: { x: 422, y: 1134 },
        dragHandle: '.drag-handle',
        deletable: false,
      },
      {
        id: mergeId,
        type: 'merge',
        data: nodeMergeDefinition({
          vectorSearchId,
          fulltextSearchId,
          graphSearchId,
        }),
        position: { x: 884, y: 524 },
        dragHandle: '.drag-handle',
        deletable: false,
      },
      {
        id: rerankId,
        type: 'rerank',
        data: nodeRerankDefinition({ docId: mergeId }),
        position: { x: 1316, y: 653 },
        dragHandle: '.drag-handle',
        deletable: false,
      },
      {
        id: llmId,
        type: 'llm',
        data: nodeLlmDefinition({ startId, docId: rerankId, botType: type }),
        position: { x: 1718, y: 470 },
        dragHandle: '.drag-handle',
        deletable: false,
      },
    ],
    edges: [
      {
        id: uniqid(),
        source: startId,
        target: vectorSearchId,
        type: 'default',
      },
      {
        id: uniqid(),
        source: startId,
        target: fulltextSearchId,
        type: 'default',
      },
      {
        id: uniqid(),
        source: startId,
        target: graphSearchId,
        type: 'default',
      },
      {
        id: uniqid(),
        source: vectorSearchId,
        target: mergeId,
        type: 'default',
      },
      {
        id: uniqid(),
        source: fulltextSearchId,
        target: mergeId,
        type: 'default',
      },
      {
        id: uniqid(),
        source: graphSearchId,
        target: mergeId,
        type: 'default',
      },
      {
        id: uniqid(),
        source: mergeId,
        target: rerankId,
        type: 'default',
      },
      {
        id: uniqid(),
        source: rerankId,
        target: llmId,
        type: 'default',
      },
    ],
  };
};

export const getInitialData = (type?: BotTypeEnum): WorkflowDefinition => {
  switch (type) {
    case 'knowledge':
      return getBotKnowledgeWorkflow(type);
    case 'common':
      return getBotCommonWorkflow(type);
    default:
      return workflow_definition;
  }
};
