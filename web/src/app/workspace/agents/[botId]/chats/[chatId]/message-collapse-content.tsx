import { Button } from '@/components/ui/button';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import { motion } from 'framer-motion';
import { ChevronRight } from 'lucide-react';

export const MessageCollapseContent = ({
  animate,
  defaultOpen,
  title,
  children,
}: {
  animate?: boolean;
  defaultOpen?: boolean;
  title: React.ReactNode;
  children: React.ReactNode;
}) => {
  return (
    <Collapsible className="group/collapsible my-2" defaultOpen={defaultOpen}>
      <motion.div
        initial={{ opacity: 0, x: 10 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{
          duration: animate ? 0.3 : 0,
          ease: 'easeIn',
        }}
      >
        <CollapsibleTrigger asChild>
          <Button variant="secondary" className="w-full cursor-pointer">
            <ChevronRight className="transition-transform duration-200 group-data-[state=open]/collapsible:rotate-90" />
            <div className="block flex-1 truncate text-left">{title}</div>
          </Button>
        </CollapsibleTrigger>
      </motion.div>
      <CollapsibleContent className="mt-2 rounded-md border p-4">
        {children}
      </CollapsibleContent>
    </Collapsible>
  );
};
