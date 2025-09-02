'use client';

import { Document, RebuildIndexesRequestIndexTypesEnum } from '@/api';
import { useCollectionContext } from '@/components/providers/collection-provider';
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
import { cn, objectKeys } from '@/lib/utils';
import { zodResolver } from '@hookform/resolvers/zod';
import { Slot } from '@radix-ui/react-slot';
import { useTranslations } from 'next-intl';
import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { toast } from 'sonner';
import { z } from 'zod';
import { DocumentIndexStatus } from './document-index-status';

const documentReBuildSchema = z.object({
  index_types: z.array(z.enum(objectKeys(RebuildIndexesRequestIndexTypesEnum))),
});

type DocumentReBuildSchemaType = z.infer<typeof documentReBuildSchema>;

export const DocumentReBuildIndex = ({
  file,
  children,
}: {
  file: Document;
  children: React.ReactNode;
}) => {
  const { collection } = useCollectionContext();
  const page_documents = useTranslations('page_documents');
  const page_collections = useTranslations('page_collections');
  const common_action = useTranslations('common.action');
  const [visible, setVisible] = useState<boolean>(false);
  const router = useRouter();
  const form = useForm<DocumentReBuildSchemaType>({
    resolver: zodResolver(documentReBuildSchema),
    defaultValues: {
      index_types: objectKeys(RebuildIndexesRequestIndexTypesEnum).filter(
        (key) => {
          const config = collection.config;
          switch (key) {
            case 'FULLTEXT':
              return config?.enable_fulltext;
            case 'GRAPH':
              return config?.enable_knowledge_graph;
            case 'SUMMARY':
              return config?.enable_summary;
            case 'VECTOR':
              return config?.enable_vector;
            case 'VISION':
              return config?.enable_vision;
          }
        },
      ),
    },
  });

  const handleRebuild = async (values: DocumentReBuildSchemaType) => {
    if (!collection.id || !file.id) return;

    if (values.index_types.length === 0) {
      toast.error(page_documents('index_rebuild_should_select_index_type'));
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
        page_documents('index_rebuild_with', {
          types: values.index_types
            .map((type) => page_collections(`index_type_${type}.title`))
            .join(', '),
        }),
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
              <DialogTitle>{page_documents('index_rebuild')}</DialogTitle>
              <DialogDescription>{file.name}</DialogDescription>
            </DialogHeader>
            <div className="my-6 flex flex-col gap-2">
              {objectKeys(RebuildIndexesRequestIndexTypesEnum)?.map((key) => {
                const config = collection.config;
                let enabled: boolean | undefined;
                switch (key) {
                  case 'FULLTEXT':
                    enabled = config?.enable_fulltext;
                    break;
                  case 'GRAPH':
                    enabled = config?.enable_knowledge_graph;
                    break;
                  case 'SUMMARY':
                    enabled = config?.enable_summary;
                    break;
                  case 'VECTOR':
                    enabled = config?.enable_vector;
                    break;
                  case 'VISION':
                    enabled = config?.enable_vision;
                    break;
                  default:
                    enabled = false;
                }
                return (
                  <FormField
                    key={key}
                    control={form.control}
                    name="index_types"
                    render={({ field }) => (
                      <FormItem>
                        <Label
                          className={cn(
                            'has-[[aria-checked=true]]:bg-accent/80 flex items-start gap-3 rounded-lg border p-3',
                            enabled
                              ? 'hover:bg-accent/50'
                              : 'cursor-not-allowed',
                          )}
                        >
                          <FormControl>
                            <Checkbox
                              checked={field.value?.includes(key)}
                              disabled={!enabled}
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
                              {page_collections(`index_type_${key}.title`)}
                              <DocumentIndexStatus
                                document={file}
                                accessorKey={
                                  key.toLowerCase() + '_index_status'
                                }
                              />
                            </div>
                            <p className="text-muted-foreground text-sm">
                              {page_collections(
                                `index_type_${key}.description`,
                              )}
                            </p>
                          </div>
                        </Label>
                      </FormItem>
                    )}
                  />
                );
              })}
            </div>
            <DialogFooter className="items-center">
              <div className="text-muted-foreground text-xs">
                {page_documents('index_rebuild_tips')}
              </div>
              <div className="ml-auto flex gap-2">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setVisible(false)}
                >
                  {common_action('cancel')}
                </Button>
                <Button type="submit">{common_action('continue')}</Button>
              </div>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
};
