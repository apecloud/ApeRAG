import { Button } from '@/components/ui/button';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import { motion } from 'framer-motion';
import { ChevronRight } from 'lucide-react';

export const MessageCollapseContent = ({
  defaultOpen,
  title,
  children,
}: {
  defaultOpen?: boolean;
  title: React.ReactNode;
  children: React.ReactNode;
}) => {
  return (
    <Collapsible className="group/collapsible my-2" defaultOpen={defaultOpen}>
      <motion.div
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{
          duration: 0.3,
          ease: 'easeIn',
        }}
      >
        <CollapsibleTrigger asChild>
          <Button variant="secondary" className="w-full cursor-pointer">
            <ChevronRight className="transition-transform duration-200 group-data-[state=open]/collapsible:rotate-90" />
            <div className="block flex-1 text-left">{title}</div>
          </Button>
        </CollapsibleTrigger>
      </motion.div>
      <CollapsibleContent className="mt-2 rounded-md border p-4">
        {children}
      </CollapsibleContent>
    </Collapsible>
  );
};
