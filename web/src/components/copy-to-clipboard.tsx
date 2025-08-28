'use client';

import { cn } from '@/lib/utils';
import copy from 'copy-to-clipboard';
import { Copy } from 'lucide-react';
import { useCallback } from 'react';
import { toast } from 'sonner';
import { Button, ButtonProps } from './ui/button';

export const CopyToClipboard = ({
  text,
  className,
  ...props
}: ButtonProps & {
  text?: string;
}) => {
  const handlerClick = useCallback(() => {
    if (text) {
      copy(text);
      toast.success('Copied');
    }
  }, [text]);

  if (!text) return;

  return (
    <Button
      size="icon"
      className={cn('cursor-pointer', className)}
      {...props}
      onClick={handlerClick}
    >
      <Copy />
    </Button>
  );
};
