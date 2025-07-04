import {
  Document,
  NodeData,
  NodeTypeEnum,
  WorkflowStyleEdgeTypeEnum,
} from '@/api';

import {
  Position,
  Edge as ReactFlowEdge,
  Node as ReactFlowNode,
} from '@xyflow/react';

export type Merge<M, N> = Omit<M, Extract<keyof M, keyof N>> & N;

export type ApeNode = Merge<
  ReactFlowNode,
  {
    data: NodeData & {
      [key in string]: unknown;
    };
    id: string;
    type: NodeTypeEnum;
  }
>;
export type ApeEdge = Merge<
  ReactFlowEdge,
  {
    type: WorkflowStyleEdgeTypeEnum;
  }
>;

export type ApeNodeHandlePosition = {
  sourcePosition?: Position;
  targetPosition?: Position;
};

export type ApeNodeConfig = {
  color?: string;
  icon?: React.ReactNode;
  label?: string;
  content?: ({ node }: { node: ApeNode }) => JSX.Element;
  width?: number;
  disableConnectionTarget?: boolean;
  disableConnectionSource?: boolean;
};

export type ApeNodesConfig = {
  [key in NodeTypeEnum]: ApeNodeConfig;
};

// to delete
export type ApeFlowNodeOutput = {
  doc_id?: string;
  rank_before?: number;
  score?: number;
  source?: string;
  text?: string;
  metadata?: {
    chunk_num?: number;
    content_ratio?: number;
    name?: string;
    object_path?: string;
    source?: string;
    recall_type?: string;
  };
};

export type ApeFlowDebugInfo = {
  event_type?:
    | 'flow_start'
    | 'flow_error'
    | 'flow_end'
    | 'node_start'
    | 'node_end'
    | 'output_chunk';
  node_id?: string;
  execution_id?: string;
  timestamp?: string;
  data?: {
    flow_id: string;
    node_type?: string;
    error?: string;
    inputs?: any;
    outputs?: {
      docs: ApeFlowNodeOutput[];
      fulltext_search_docs: ApeFlowNodeOutput[];
      vector_search_docs: ApeFlowNodeOutput[];
    };
  };
};

export type ApeFlowStatus = 'running' | 'completed' | 'stopped';
export type ApeFlowNodeStatus = 'pending' | 'running' | 'complated' | 'stopped';

/**
 * collection
 */
export type CollectionConfigSource =
  | 'system'
  | 'local'
  | 's3'
  | 'oss'
  | 'ftp'
  | 'url'
  | 'git'
  | 'email'
  | 'feishu';

export type CollectionEmailSource = 'gmail' | 'outlook' | 'qqmail' | 'others';

/**
 * documents
 */
export type ApeDocumentConfig = {
  path?: string;
  labels?: {
    [key in string]: string;
  }[];
};

export type ApeDocument = Merge<Document, { config?: ApeDocumentConfig }>;

/**
 * chunks
 */
export interface Chunk {
  id: string;
  text: string;
  metadata?: {
    pdf_source_map?: {
      page_idx: number;
      bbox: [number, number, number, number];
    }[];
    md_source_map?: [number, number];
    [key: string]: any;
  };
}
