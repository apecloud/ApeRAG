'use client';
import {
  Drawer,
  DrawerContent,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
} from '@/components/ui/drawer';
import type { AuditLog } from '@/features/audit/types';
import { useFormatter, useTranslations } from 'next-intl';
import dynamic from 'next/dynamic';
import { useMemo } from 'react';

// Markdown 携带 remark/rehype/highlight 等共享大包. 只有 drawer 打开后才需要
// 渲染 request_data / response_data 的 JSON 代码块, 拆 dynamic chunk 把
// 它从 audit-logs 路由首屏 bundle 中移走.
const Markdown = dynamic(
  () => import('@/components/markdown').then((m) => ({ default: m.Markdown })),
  { ssr: false, loading: () => null },
);

export const AuditLogDetail = ({
  auditLog,
  children,
}: {
  auditLog: AuditLog;
  children: React.ReactNode;
}) => {
  const format = useFormatter();
  const page_audit_logs = useTranslations('page_audit_logs');

  const responseData = useMemo(() => {
    const res = auditLog.response_data || '{}';
    let result = res;
    try {
      result =
        '``` json\n' + JSON.stringify(JSON.parse(res), undefined, 2) + '\n```';
    } catch (err) {
      console.log(err);
    }
    return result;
  }, [auditLog.response_data]);
  const requestData = useMemo(() => {
    const request = auditLog.request_data || '{}';
    let result = request;
    try {
      result =
        '``` json\n' +
        JSON.stringify(JSON.parse(request), undefined, 2) +
        '\n```';
    } catch (err) {
      console.log(err);
    }
    return result;
  }, [auditLog.request_data]);

  return (
    <>
      <Drawer direction="right" handleOnly={true}>
        <DrawerTrigger asChild>{children}</DrawerTrigger>
        <DrawerContent className="border-border/70 bg-card flex border-l data-[vaul-drawer-direction=right]:w-[calc(100vw-1rem)] data-[vaul-drawer-direction=right]:sm:w-[32rem] data-[vaul-drawer-direction=right]:sm:max-w-none data-[vaul-drawer-direction=right]:md:w-[40rem] data-[vaul-drawer-direction=right]:lg:w-[48rem]">
          <DrawerHeader className="border-border/70 border-b">
            <DrawerTitle className="font-serif text-2xl font-normal">
              {page_audit_logs('metadata.title')}
            </DrawerTitle>
          </DrawerHeader>
          <div className="flex select-text flex-col gap-4 overflow-auto p-4 text-sm">
            <div className="border-border/70 bg-muted rounded-xl border p-3">
              <div className="text-muted-foreground font-mono text-[11px] uppercase">
                {page_audit_logs('detail.user_agent')}
              </div>
              <div className="mt-1 break-words">{auditLog.user_agent}</div>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <AuditMeta
                label={page_audit_logs('detail.ip')}
                value={auditLog.ip_address}
              />
              <AuditMeta
                label={page_audit_logs('detail.user_id')}
                value={auditLog.user_id}
              />
              <AuditMeta
                label={page_audit_logs('detail.request_id')}
                value={auditLog.request_id}
              />
              <AuditMeta
                label={page_audit_logs('detail.api')}
                value={auditLog.api_name}
              />
              <AuditMeta
                label={page_audit_logs('detail.path')}
                value={auditLog.path}
              />
              <AuditMeta
                label={page_audit_logs('detail.method')}
                value={auditLog.http_method}
              />
              <AuditMeta
                label={page_audit_logs('detail.status_code')}
                value={String(auditLog.status_code || '--')}
              />
              <AuditMeta
                label={page_audit_logs('detail.resource_type')}
                value={auditLog.resource_type}
              />
            </div>

            <div>
              <div className="text-muted-foreground -mb-3 flex flex-col gap-1 sm:flex-row sm:justify-between">
                <div>{page_audit_logs('detail.request_data')}</div>
                <div>
                  {auditLog.start_time
                    ? format.dateTime(new Date(auditLog.start_time), 'long')
                    : ''}
                </div>
              </div>
              <Markdown>{requestData}</Markdown>
            </div>

            <div>
              <div className="text-muted-foreground -mb-3 flex flex-col gap-1 sm:flex-row sm:justify-between">
                <div>{page_audit_logs('detail.response_data')}</div>
                <div>
                  {auditLog.end_time
                    ? format.dateTime(new Date(auditLog.end_time), 'long')
                    : ''}
                </div>
              </div>
              <Markdown>{responseData}</Markdown>
            </div>

            <div className="border-border/70 bg-muted rounded-xl border p-3">
              <div className="text-muted-foreground font-mono text-[11px] uppercase">
                {page_audit_logs('detail.error_messages')}
              </div>
              <div>{auditLog.error_message || '--'}</div>
            </div>

            <AuditMeta
              label={page_audit_logs('detail.resource_id')}
              value={auditLog.resource_id}
            />
          </div>
        </DrawerContent>
      </Drawer>
    </>
  );
};

const AuditMeta = ({
  label,
  value,
}: {
  label: string;
  value?: string | number | null;
}) => (
  <div className="border-border/70 bg-muted rounded-xl border p-3">
    <div className="text-muted-foreground font-mono text-[11px] uppercase">
      {label}
    </div>
    <div className="mt-1 break-words font-mono text-xs">{value || '--'}</div>
  </div>
);
