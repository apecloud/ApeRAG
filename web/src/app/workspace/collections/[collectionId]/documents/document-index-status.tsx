import type {
  Document,
  DocumentIndexStatus as DocumentIndexStatusType,
} from '@/features/document/types';
import { cn } from '@/lib/utils';
import _ from 'lodash';

const getIndexStatusBg = (status?: DocumentIndexStatusType | null) => {
  const data: Record<DocumentIndexStatusType, string> = {
    ACTIVE: 'bg-green-500',
    RUNNING: 'bg-sky-500',
    FAILED: 'bg-red-500',
    PENDING: 'bg-amber-500',
  };
  return status ? data[status] : 'bg-gray-500';
};

export const DocumentIndexStatus = ({
  document,
  accessorKey,
}: {
  document: Document;
  accessorKey: string;
}) => {
  const status = _.get(document, accessorKey);
  const color = getIndexStatusBg(status);
  return (
    <div className="flex flex-row items-center gap-2">
      <div className={cn('size-1.5 rounded-4xl', color)}></div>
      <div className="text-xs">{_.capitalize(status)}</div>
    </div>
  );
};
