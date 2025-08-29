'use client';

import { cn } from '@/lib/utils';
import copy from 'copy-to-clipboard';
import _ from 'lodash';
import { Copy } from 'lucide-react';
import { useTranslations } from 'next-intl';
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
  const components_copy_to_clipboard = useTranslations(
    'components.copy_to_clipboard',
  );

  const handlerClick = useCallback(() => {
    if (text) {
      copy(text);
      toast.success(components_copy_to_clipboard('copied'));
    }
  }, [components_copy_to_clipboard, text]);

  return (
    <Button
      size="icon"
      disabled={_.isEmpty(text)}
      className={cn('cursor-pointer', className)}
      {...props}
      onClick={handlerClick}
    >
      <Copy />
    </Button>
  );
};
