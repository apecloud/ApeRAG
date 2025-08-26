'use client';
import { AuditLog } from '@/api';
import { Markdown } from '@/components/markdown';
import {
  Drawer,
  DrawerContent,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
} from '@/components/ui/drawer';
import { useFormatter } from 'next-intl';
import { useMemo } from 'react';

export const AuditLogDetail = ({
  auditLog,
  children,
}: {
  auditLog: AuditLog;
  children: React.ReactNode;
}) => {
  const format = useFormatter();

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

  console.log(auditLog.response_data || 'asdasdasd');

  return (
    <>
      <Drawer direction="right" handleOnly={true}>
        <DrawerTrigger asChild>{children}</DrawerTrigger>
        <DrawerContent className="flex sm:min-w-lg md:min-w-xl lg:min-w-2xl">
          <DrawerHeader>
            <DrawerTitle className="font-bold">Audit Log</DrawerTitle>
          </DrawerHeader>
          <div className="flex flex-col gap-4 overflow-auto p-4 text-sm select-text">
            <div>
              <div className="text-muted-foreground">User Agent:</div>
              <div>{auditLog.user_agent}</div>
            </div>

            <div>
              <div className="text-muted-foreground">IP:</div>
              <div>{auditLog.ip_address}</div>
            </div>

            <div>
              <div className="text-muted-foreground">User ID:</div>
              <div>{auditLog.user_id}</div>
            </div>

            <div>
              <div className="text-muted-foreground">Request ID:</div>
              <div>{auditLog.request_id}</div>
            </div>

            <div>
              <div className="text-muted-foreground">API:</div>
              <div>{auditLog.api_name}</div>
            </div>

            <div>
              <div className="text-muted-foreground">Path:</div>
              <div>{auditLog.path}</div>
            </div>

            <div>
              <div className="text-muted-foreground">Method:</div>
              <div>{auditLog.http_method}</div>
            </div>

            <div>
              <div className="text-muted-foreground">Status Code:</div>
              <div>{auditLog.status_code}</div>
            </div>

            <div>
              <div className="text-muted-foreground -mb-3 flex justify-between">
                <div>Request Data:</div>
                <div>
                  {auditLog.start_time
                    ? format.dateTime(new Date(auditLog.start_time), 'long')
                    : ''}
                </div>
              </div>
              <Markdown>
                {'``` json\n' +
                  JSON.stringify(
                    JSON.parse(auditLog.request_data || ''),
                    undefined,
                    2,
                  ) +
                  '\n```'}
              </Markdown>
            </div>

            <div>
              <div className="text-muted-foreground -mb-3 flex justify-between">
                <div>Response Data:</div>
                <div>
                  {auditLog.end_time
                    ? format.dateTime(new Date(auditLog.end_time), 'long')
                    : ''}
                </div>
              </div>
              <Markdown>{responseData}</Markdown>
            </div>

            <div>
              <div className="text-muted-foreground">Error Messages:</div>
              <div>{auditLog.error_message || '--'}</div>
            </div>

            <div>
              <div className="text-muted-foreground">Resource ID:</div>
              <div>{auditLog.resource_id || '--'}</div>
            </div>

            <div>
              <div className="text-muted-foreground">Resource Type:</div>
              <div>{auditLog.resource_type}</div>
            </div>
          </div>
        </DrawerContent>
      </Drawer>
    </>
  );
};
