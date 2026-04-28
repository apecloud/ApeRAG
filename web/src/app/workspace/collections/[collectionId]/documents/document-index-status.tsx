import type {
  Document,
  DocumentIndexStatus as DocumentIndexStatusType,
} from '@/features/document/types';
import { cn } from '@/lib/utils';
import _ from 'lodash';
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
  const status = _.get(document, accessorKey) as
    | DocumentIndexStatusType
    | null
    | undefined;
  // Localized label per status; null/undefined (index row not yet
  // created — typical for freshly-uploaded PENDING documents whose
  // indexing pipeline hasn't started) maps to ``not_started`` so the
  // cell never renders an empty string next to a gray dot (the prior
  // shape was visually indistinguishable from a broken cell —
  // earayu2 bug report msg=dbe6f513 task #10).
  // Explicit map (vs ``\`index_status_${status.toLowerCase()}\``) so
  // ``next-intl``'s typed-key checker can verify each i18n key at
  // compile time — a dynamic template would widen to the full
  // namespace union and require placeholder args for keys it never
  // resolves to.
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
