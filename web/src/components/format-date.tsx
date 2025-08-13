'use client';

export type FormatDateProps = { datetime?: Date };
import { useFormatter, useNow } from 'next-intl';
import { Tooltip, TooltipTrigger } from './ui/tooltip';

export const FormatDate = ({ datetime }: FormatDateProps): React.ReactNode => {
  const format = useFormatter();
  const now = useNow();
  if (!datetime) return '--';

  return (
    <Tooltip>
      <TooltipTrigger>{format.relativeTime(datetime, now)}</TooltipTrigger>
      {/* <TooltipContent>{format.dateTime(datetime, 'medium')}</TooltipContent> */}
    </Tooltip>
  );
};
