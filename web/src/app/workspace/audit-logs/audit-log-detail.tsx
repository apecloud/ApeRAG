'use client';
import { AuditLog } from '@/api';
import { Markdown } from '@/components/markdown';
import { Button } from '@/components/ui/button';
import {
  Drawer,
  DrawerContent,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
} from '@/components/ui/drawer';
import { ScanEye } from 'lucide-react';
import { useFormatter } from 'next-intl';

export const AuditLogDetail = ({ auditLog }: { auditLog: AuditLog }) => {
  const format = useFormatter();
  return (
    <>
      <Drawer direction="right" handleOnly={true}>
        <DrawerTrigger asChild>
          <Button variant="ghost" size="icon">
            <ScanEye />
          </Button>
        </DrawerTrigger>
        <DrawerContent className="flex min-w-xl">
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
              <Markdown>
                {'``` json\n' +
                  JSON.stringify(
                    JSON.parse(auditLog.response_data || ''),
                    undefined,
                    2,
                  ) +
                  '\n```'}
              </Markdown>
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
