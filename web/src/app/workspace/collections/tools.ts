import { DocumentStatusEnum } from '@/api';

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
