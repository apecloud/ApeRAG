import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
} from '@/components/ui/form';
import { Textarea } from '@/components/ui/textarea';
import { Slot } from '@radix-ui/react-slot';
import { ReactNode, useState } from 'react';
import { useForm } from 'react-hook-form';

export const SearchTest = ({ children }: { children: ReactNode }) => {
  const [visible, setVisible] = useState<boolean>(false);
  const form = useForm();

  const handleSubmit = () => {};

  return (
    <Dialog open={visible} onOpenChange={() => setVisible(false)}>
      <DialogTrigger asChild>
        <Slot
          onClick={(e) => {
            setVisible(true);
            e.preventDefault();
          }}
        >
          {children}
        </Slot>
      </DialogTrigger>
      <DialogContent>
        <Form {...form}>
          <form
            onSubmit={form.handleSubmit(handleSubmit)}
            className="space-y-8"
          >
            <DialogHeader>
              <DialogTitle>Search test</DialogTitle>
              <DialogDescription></DialogDescription>
            </DialogHeader>
            <FormField
              control={form.control}
              name="label"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Query</FormLabel>
                  <FormControl>
                    <Textarea placeholder="Enter your question." {...field} />
                  </FormControl>
                </FormItem>
              )}
            />

            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setVisible(false)}
              >
                Cancel
              </Button>
              <Button type="submit">Test</Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
};
