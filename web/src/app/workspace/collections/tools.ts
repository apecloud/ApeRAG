import type { DocumentStatus } from '@/features/document/types';

export const getDocumentStatusColor = (status?: DocumentStatus | null) => {
  const data: Record<DocumentStatus, string> = {
    PENDING: 'text-muted-foreground',
    RUNNING: 'text-muted-foreground',
    COMPLETE: 'text-accent-foreground',
    UPLOADED: 'text-muted-foreground',
    FAILED: 'text-red-500',
    EXPIRED: 'text-muted-foreground line-through',
    DELETED: 'text-muted-foreground line-through',
    DELETING: 'text-muted-foreground line-through',
  };
  return status ? data[status] : 'text-muted-foreground';
};
