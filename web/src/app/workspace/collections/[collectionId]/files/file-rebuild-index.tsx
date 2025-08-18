'use client';

import { Collection, Document } from '@/api';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Form, FormControl, FormField, FormItem } from '@/components/ui/form';
import { Label } from '@/components/ui/label';
import { apiClient } from '@/lib/api/client';
import { FileIndexType } from '@/lib/document';
import { objectKeys } from '@/lib/utils';
import { zodResolver } from '@hookform/resolvers/zod';
import { Slot } from '@radix-ui/react-slot';
import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { toast } from 'sonner';
import { z } from 'zod';
import { FileIndexStatus } from './file-index-status';

const fileReBuildSchema = z.object({
  index_types: z.array(z.enum(objectKeys(FileIndexType))),
});

type FileReBuildSchemaType = z.infer<typeof fileReBuildSchema>;

export const FileReBuildIndex = ({
  collection,
  file,
  children,
}: {
  collection: Collection;
  file: Document;
  children: React.ReactNode;
}) => {
  const [visible, setVisible] = useState<boolean>(false);
  const router = useRouter();
  const form = useForm<FileReBuildSchemaType>({
    resolver: zodResolver(fileReBuildSchema),
    defaultValues: {
      index_types: objectKeys(FileIndexType),
    },
  });

  const handleRebuild = async (values: FileReBuildSchemaType) => {
    if (!collection.id || !file.id) return;

    if (values.index_types.length === 0) {
      toast.error('You have to select at least one item.');
      return;
    }

    const res =
      await apiClient.defaultApi.collectionsCollectionIdDocumentsDocumentIdRebuildIndexesPost(
        {
          collectionId: collection.id,
          documentId: file.id,
          rebuildIndexesRequest: {
            index_types: values.index_types,
          },
        },
      );

    if (res.status === 200) {
      toast.success(
        `Index rebuild initiated for types: ${values.index_types.join(', ')}`,
      );
      setVisible(false);
      setTimeout(router.refresh, 300);
    }
  };

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
          <form onSubmit={form.handleSubmit(handleRebuild)}>
            <DialogHeader>
              <DialogTitle>Rebuild File Index</DialogTitle>
              <DialogDescription>{file.name}</DialogDescription>
            </DialogHeader>
            <div className="my-6 flex flex-col gap-4">
              {objectKeys(FileIndexType)?.map((key) => {
                const item = FileIndexType[key];
                return (
                  <FormField
                    key={key}
                    control={form.control}
                    name="index_types"
                    render={({ field }) => (
                      <FormItem>
                        <Label className="hover:bg-accent/50 has-[[aria-checked=true]]:bg-accent/80 flex items-start gap-3 rounded-lg border p-3">
                          <FormControl>
                            <Checkbox
                              checked={field.value?.includes(key)}
                              onCheckedChange={(checked) => {
                                return checked
                                  ? field.onChange([...field.value, key])
                                  : field.onChange(
                                      field.value?.filter(
                                        (value) => value !== key,
                                      ),
                                    );
                              }}
                            />
                          </FormControl>
                          <div className="grid gap-1.5 font-normal">
                            <div className="item-center flex flex-row justify-between text-sm leading-none font-medium">
                              {item.title}
                              <FileIndexStatus
                                document={file}
                                accessorKey={
                                  key.toLowerCase() + '_index_status'
                                }
                              />
                            </div>
                            <p className="text-muted-foreground text-sm">
                              {item.description}
                            </p>
                          </div>
                        </Label>
                      </FormItem>
                    )}
                  />
                );
              })}
            </div>

            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setVisible(false)}
              >
                Cancel
              </Button>
              <Button type="submit">Save</Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
};
