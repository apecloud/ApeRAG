'use client';

import { FetchUrlResultItem, FetchUrlResultItemFetchStatusEnum } from '@/api';
import { useCollectionContext } from '@/components/providers/collection-provider';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { apiClient } from '@/lib/api/client';
import { AlertCircle, CheckCircle2, Globe, LoaderCircle } from 'lucide-react';
import { useTranslations } from 'next-intl';
import { useCallback, useState } from 'react';

type UrlImportResult = {
  url: string;
  fetch_status: 'success' | 'error';
  document_id?: string;
  filename?: string;
  size?: number;
  status?: string;
  error?: string;
};

/** Returns "hostname/first-path-segment" as a short human-readable label */
function shortUrl(url: string): string {
  try {
    const u = new URL(url);
    const path = u.pathname.replace(/\/$/, '').split('/')[1];
    return path ? `${u.hostname}/${path}` : u.hostname;
  } catch {
    return url.slice(0, 40);
  }
}

function ResultSummary({ results }: { results: UrlImportResult[] }) {
  const succeeded = results.filter((r) => r.fetch_status === 'success');
  const failed = results.filter((r) => r.fetch_status === 'error');

  return (
    <div className="space-y-2 text-sm">
      {/* Counts */}
      <div className="flex flex-wrap gap-3">
        {succeeded.length > 0 && (
          <span className="flex items-center gap-1 text-emerald-600">
            <CheckCircle2 className="size-3.5 shrink-0" />
            {succeeded.length} 个成功
          </span>
        )}
        {failed.length > 0 && (
          <span className="flex items-center gap-1 text-red-500">
            <AlertCircle className="size-3.5 shrink-0" />
            {failed.length} 个失败
          </span>
        )}
      </div>

      {/* Error details */}
      {failed.length > 0 && (
        <div className="max-h-36 space-y-1.5 overflow-y-auto rounded-md border border-red-200 bg-red-50 p-3 dark:border-red-900 dark:bg-red-950/30">
          {failed.map((r, i) => (
            <div key={i} className="text-xs">
              <span
                className="font-medium text-red-600 dark:text-red-400"
                title={r.url}
              >
                {shortUrl(r.url)}
              </span>
              {r.error && (
                <p className="text-muted-foreground mt-0.5 leading-relaxed">
                  {r.error}
                </p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

type Props = {
  onSuccess: (results: UrlImportResult[]) => void;
};

export const UrlImport = ({ onSuccess }: Props) => {
  const { collection } = useCollectionContext();
  const t = useTranslations('page_documents');
  const [urlText, setUrlText] = useState('');
  const [isFetching, setIsFetching] = useState(false);
  const [results, setResults] = useState<UrlImportResult[] | null>(null);

  const parseUrls = (text: string): string[] =>
    text
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line.startsWith('http://') || line.startsWith('https://'));

  const handleFetch = useCallback(async () => {
    if (!collection.id) return;
    const urls = parseUrls(urlText);
    if (urls.length === 0) return;

    setIsFetching(true);
    setResults(null);

    try {
      const res = await apiClient.defaultApi.collectionsCollectionIdDocumentsFetchUrlPost(
        {
          collectionId: collection.id,
          fetchUrlRequest: { urls },
        },
        { timeout: 1000 * 60 },
      );

      const fetchResults: UrlImportResult[] = (res.data.results as FetchUrlResultItem[]).map(
        (item) => ({
          url: item.url,
          fetch_status:
            item.fetch_status === FetchUrlResultItemFetchStatusEnum.success
              ? ('success' as const)
              : ('error' as const),
          document_id: item.document_id ?? undefined,
          filename: item.filename ?? undefined,
          size: item.size ?? undefined,
          status: item.status ?? undefined,
          error: item.error ?? undefined,
        }),
      );
      setResults(fetchResults);

      const succeeded = fetchResults.filter((r) => r.fetch_status === 'success');
      if (succeeded.length > 0) {
        onSuccess(fetchResults);
      }
    } catch {
      setResults(
        parseUrls(urlText).map((url) => ({
          url,
          fetch_status: 'error' as const,
          error: 'Request failed',
        })),
      );
    } finally {
      setIsFetching(false);
    }
  }, [collection.id, urlText, onSuccess]);

  const urls = parseUrls(urlText);
  const isValid = urls.length > 0 && urls.length <= 10;

  return (
    <div className="flex flex-col gap-4">
      <p className="text-muted-foreground text-sm">{t('import_url_desc')}</p>

      <Textarea
        placeholder={t('import_url_placeholder')}
        value={urlText}
        onChange={(e) => setUrlText(e.target.value)}
        rows={4}
        disabled={isFetching}
        className="resize-none font-mono text-sm"
      />

      {urls.length > 10 && (
        <p className="text-sm text-red-500">
          {t('import_url_too_many', { count: String(urls.length) })}
        </p>
      )}

      <ul className="text-muted-foreground space-y-1 text-xs">
        <li>• {t('import_url_tips_1')}</li>
        <li>• {t('import_url_tips_2')}</li>
        <li>• {t('import_url_tips_3')}</li>
      </ul>

      {results && (
        <ResultSummary results={results} />
      )}

      <div className="flex justify-end">
        <Button onClick={handleFetch} disabled={!isValid || isFetching} className="min-w-28">
          {isFetching ? (
            <>
              <LoaderCircle className="animate-spin" />
              {t('import_url_fetching')}
            </>
          ) : (
            <>
              <Globe />
              {t('import_url_btn')}
            </>
          )}
        </Button>
      </div>
    </div>
  );
};
