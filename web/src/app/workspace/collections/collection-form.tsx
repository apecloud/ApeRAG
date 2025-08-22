'use client';

import { Collection } from '@/api';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Form } from '@/components/ui/form';
import { apiClient } from '@/lib/api/client';
import { zodResolver } from '@hookform/resolvers/zod';
import { useRouter } from 'next/navigation';
import { useCallback } from 'react';
import { useForm } from 'react-hook-form';
import { toast } from 'sonner';
import * as z from 'zod';

const defaultValue = {
  label: '',
  base_url: '',
  completion_dialect: 'openai',
  embedding_dialect: 'openai',
  rerank_dialect: 'jina_ai',
};

const collectionSchema = z.object({
  title: z.string().min(1),
  description: z.string(),
  config: z.object({
    source: z.string('system'),
    index_types: z.array(
      z.enum(['vector', 'fulltext', 'graph', 'summary', 'vision']),
    ),
    enable_fulltext: z.boolean(),
    enable_knowledge_graph: z.boolean(),
    enable_summary: z.boolean(),
    enable_vector: z.boolean(),
    enable_vision: z.boolean(),
    completion: z.object({
      custom_llm_provider: z.string(),
      model: z.string(),
      model_service_provider: z.string(),
    }),
    embedding: z.object({
      custom_llm_provider: z.string(),
      model: z.string(),
      model_service_provider: z.string(),
    }),
  }),
});

export const CollectionForm = ({
  collection,
  action,
}: {
  collection?: Collection;
  action: 'add' | 'edit';
}) => {
  const router = useRouter();
  const form = useForm<z.infer<typeof collectionSchema>>({
    resolver: zodResolver(collectionSchema),
    defaultValues: { ...defaultValue, ...collection },
  });

  const handleCreateOrUpdate = useCallback(
    async (values: z.infer<typeof collectionSchema>) => {
      const { data: params, error } = collectionSchema.safeParse(values);
      if (error) {
        return;
      }

      if (action === 'edit') {
        if (!collection?.id) return;
        const res = await apiClient.defaultApi.collectionsCollectionIdPut({
          collectionId: collection.id,
          collectionUpdate: params,
        });

        if (res.status === 200) {
          toast.success('Saved successfully.');
        }
      }
      if (action === 'add') {
        const res = await apiClient.defaultApi.collectionsPost({
          collectionCreate: params,
        });

        if (res.status === 200) {
          toast.success('Saved successfully.');
          router.push('/workspace/collections');
        }
      }
    },
    [action, collection?.id, router],
  );

  return (
    <>
      <Form {...form}>
        <form
          onSubmit={form.handleSubmit(handleCreateOrUpdate)}
          className="flex flex-col gap-4"
        >
          <Card>
            <CardHeader>
              <CardTitle>General</CardTitle>
              <CardDescription></CardDescription>
            </CardHeader>
            <CardContent></CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Index Types</CardTitle>
              <CardDescription>
                Select the AI capabilities you need, we will build corresponding
                indexes for your documents
              </CardDescription>
            </CardHeader>
            <CardContent></CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Model Settings</CardTitle>
              <CardDescription>
                Select AI models for document processing. Different index types
                require different model support
              </CardDescription>
            </CardHeader>
            <CardContent></CardContent>
          </Card>

          <div className="flex justify-end">
            <Button type="submit" className="px-12">
              Save
            </Button>
          </div>
        </form>
      </Form>
    </>
  );
};
