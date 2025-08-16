'use client';

import copy from 'copy-to-clipboard';
import { Copy } from 'lucide-react';
import { useCallback } from 'react';
import { toast } from 'sonner';
import { Button } from './ui/button';

export const CopyToClipboard = ({
  text,
  ...props
}: React.ComponentProps<'button'> & {
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
    <Button {...props} onClick={handlerClick}>
      <Copy />
    </Button>
  );
};
