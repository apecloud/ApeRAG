import type {
  Document,
  DocumentIndexStatus as DocumentIndexStatusType,
} from '@/features/document/types';
import { cn } from '@/lib/utils';
import { useTranslations } from 'next-intl';

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
  const page_documents = useTranslations('page_documents');
  const status = document[accessorKey as keyof Document] as
    | DocumentIndexStatusType
    | null
    | undefined;
  // See workspace mirror (../../../workspace/.../document-index-status.tsx)
  // for the rationale — null/undefined index_status maps to localized
  // "not started" so freshly-uploaded PENDING documents don't render
  // an empty cell next to a gray dot. Explicit switch keeps each
  // i18n key compile-time verifiable by ``next-intl``.
  let label: string;
  switch (status) {
    case 'ACTIVE':
      label = page_documents('index_status_active');
      break;
    case 'RUNNING':
      label = page_documents('index_status_running');
      break;
    case 'PENDING':
      label = page_documents('index_status_pending');
      break;
    case 'FAILED':
      label = page_documents('index_status_failed');
      break;
    default:
      label = page_documents('index_status_not_started');
  }
  const color = getIndexStatusBg(status);
  return (
    <div className="flex flex-row items-center gap-2">
      <div className={cn('size-1.5 rounded-4xl', color)}></div>
      <div className={cn('text-xs', !status && 'text-muted-foreground')}>
        {label}
      </div>
    </div>
  );
};
