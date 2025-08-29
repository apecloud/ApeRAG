import { DocumentStatusEnum, RebuildIndexesRequestIndexTypesEnum } from '@/api';

export const FileIndexTypes: {
  [key in RebuildIndexesRequestIndexTypesEnum]: {
    title: string;
    description: string;
  };
} = {
  VECTOR: {
    title: 'Vector',
    description:
      'Represents data (text, images, etc.) as high-dimensional vectors (embeddings) to enable similarity-based retrieval (e.g., cosine similarity).',
  },
  FULLTEXT: {
    title: 'Fulltext',
    description:
      'Indexes keywords within text using techniques like inverted indexing for exact or fuzzy',
  },
  GRAPH: {
    title: 'Graph',
    description:
      'Stores data as nodes (entities) and edges (relationships), enabling complex relational',
  },
  SUMMARY: {
    title: 'Summary',
    description:
      'Indexes summaries or metadata of long documents instead of raw content for faster',
  },
  VISION: {
    title: 'Vision',
    description:
      'Encodes visual content (images/videos) into feature vectors for content-based retrieval.',
  },
};

export const getDocumentStatusColor = (status?: DocumentStatusEnum) => {
  const data: {
    [key in DocumentStatusEnum]: string;
  } = {
    [DocumentStatusEnum.PENDING]: 'text-muted-foreground',
    [DocumentStatusEnum.RUNNING]: 'text-muted-foreground',
    [DocumentStatusEnum.COMPLETE]: 'text-accent-foreground',
    [DocumentStatusEnum.UPLOADED]: 'text-muted-foreground',
    [DocumentStatusEnum.FAILED]: 'text-red-500',
    [DocumentStatusEnum.EXPIRED]: 'text-muted-foreground line-through',
    [DocumentStatusEnum.DELETED]: 'text-muted-foreground line-through',
    [DocumentStatusEnum.DELETING]: 'text-muted-foreground line-through',
  };
  return status ? data[status] : 'text-muted-foreground';
};
