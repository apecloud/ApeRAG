'use client';

import copy from 'copy-to-clipboard';
import { Copy } from 'lucide-react';
import { useCallback } from 'react';
import { toast } from 'sonner';
import { Button, ButtonProps } from './ui/button';

export const CopyToClipboard = ({
  text,
  ...props
}: ButtonProps & {
  text?: string;
}) => {
  const handlerClick = useCallback(() => {
    if (text) {
      copy(text);
      toast.success('copied');
    }
  }, [text]);

  if (!text) return;

  return (
    <Button size="icon" {...props} onClick={handlerClick}>
      <Copy />
    </Button>
  );
};
