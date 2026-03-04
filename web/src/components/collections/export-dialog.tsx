'use client';

import { ExportTaskResponse } from '@/api';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Progress } from '@/components/ui/progress';
import { apiClient } from '@/lib/api/client';
import { Slot } from '@radix-ui/react-slot';
import { useTranslations } from 'next-intl';
import { useCallback, useEffect, useRef, useState } from 'react';
import { toast } from 'sonner';

type ExportStep = 'confirm' | 'processing' | 'completed' | 'failed';

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

export const CollectionExport = ({
  collectionId,
  children,
}: {
  collectionId: string;
  children?: React.ReactNode;
}) => {
  const [open, setOpen] = useState(false);
  const [step, setStep] = useState<ExportStep>('confirm');
  const [taskStatus, setTaskStatus] = useState<ExportTaskResponse | null>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const t = useTranslations('page_collections');

  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  }, []);

  const startPolling = useCallback(
    (taskId: string) => {
      stopPolling();
      pollingRef.current = setInterval(async () => {
        try {
          const res = await apiClient.defaultApi.getExportTask({ taskId });
          const data = res.data;
          setTaskStatus(data);

          if (data.status === 'COMPLETED') {
            stopPolling();
            setStep('completed');
          } else if (data.status === 'FAILED') {
            stopPolling();
            setStep('failed');
          }
        } catch {
          // polling errors are non-fatal
        }
      }, 2000);
    },
    [stopPolling],
  );

  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  const handleOpen = useCallback(() => {
    setStep('confirm');
    setTaskStatus(null);
    setOpen(true);
  }, []);

  const handleClose = useCallback(() => {
    if (step === 'processing') return;
    stopPolling();
    setOpen(false);
  }, [step, stopPolling]);

  const handleStartExport = useCallback(async () => {
    try {
      const res = await apiClient.defaultApi.createExportTask({ collectionId });
      const data = res.data;
      setTaskStatus(data);
      setStep('processing');
      startPolling(data.export_task_id);
    } catch (err: unknown) {
      const status = (err as { response?: { status?: number } })?.response?.status;
      if (status === 429) {
        toast.error(t('export_knowledge_base_too_many_tasks'));
      } else {
        const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data
          ?.detail;
        toast.error(detail || t('export_knowledge_base_failed_description'));
      }
      setOpen(false);
    }
  }, [collectionId, startPolling, t]);

  const handleRetry = useCallback(() => {
    setStep('confirm');
    setTaskStatus(null);
  }, []);

  const handleDownload = useCallback(() => {
    if (!taskStatus?.download_url) return;
    const a = document.createElement('a');
    a.href = taskStatus.download_url;
    a.click();
  }, [taskStatus]);

  return (
    <>
      <Slot
        onClick={(e) => {
          handleOpen();
          e.preventDefault();
        }}
      >
        {children}
      </Slot>

      <Dialog open={open} onOpenChange={handleClose}>
        <DialogContent
          onInteractOutside={(e) => {
            if (step === 'processing') e.preventDefault();
          }}
          onEscapeKeyDown={(e) => {
            if (step === 'processing') e.preventDefault();
          }}
        >
          {step === 'confirm' && (
            <>
              <DialogHeader>
                <DialogTitle>{t('export_knowledge_base_confirm_title')}</DialogTitle>
                <DialogDescription>
                  {t('export_knowledge_base_confirm_content')}
                </DialogDescription>
              </DialogHeader>
              <DialogFooter>
                <Button variant="outline" onClick={handleClose}>
                  Cancel
                </Button>
                <Button onClick={handleStartExport}>
                  {t('export_knowledge_base_start')}
                </Button>
              </DialogFooter>
            </>
          )}

          {step === 'processing' && (
            <>
              <DialogHeader>
                <DialogTitle>{t('export_knowledge_base_processing_title')}</DialogTitle>
                <DialogDescription>
                  {taskStatus?.message ?? '...'}
                </DialogDescription>
              </DialogHeader>
              <div className="py-4">
                <Progress value={taskStatus?.progress ?? 0} className="w-full" />
                <p className="text-muted-foreground mt-2 text-sm text-center">
                  {taskStatus?.progress ?? 0}%
                </p>
              </div>
            </>
          )}

          {step === 'completed' && (
            <>
              <DialogHeader>
                <DialogTitle>{t('export_knowledge_base_completed_title')}</DialogTitle>
                <DialogDescription>
                  {t('export_knowledge_base_completed_description', {
                    size: taskStatus?.file_size
                      ? formatFileSize(taskStatus.file_size)
                      : '—',
                  })}
                </DialogDescription>
              </DialogHeader>
              <DialogFooter>
                <Button variant="outline" onClick={handleClose}>
                  Cancel
                </Button>
                <Button onClick={handleDownload}>
                  {t('export_knowledge_base_download')}
                </Button>
              </DialogFooter>
            </>
          )}

          {step === 'failed' && (
            <>
              <DialogHeader>
                <DialogTitle>{t('export_knowledge_base_failed_title')}</DialogTitle>
                <DialogDescription>
                  {taskStatus?.error_message ||
                    t('export_knowledge_base_failed_description')}
                </DialogDescription>
              </DialogHeader>
              <DialogFooter>
                <Button variant="outline" onClick={handleClose}>
                  Cancel
                </Button>
                <Button onClick={handleRetry}>
                  {t('export_knowledge_base_retry')}
                </Button>
              </DialogFooter>
            </>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
};
