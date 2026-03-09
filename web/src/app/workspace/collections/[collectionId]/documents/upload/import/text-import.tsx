'use client';

import { useCollectionContext } from '@/components/providers/collection-provider';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { apiClient } from '@/lib/api/client';
import { LoaderCircle, Type } from 'lucide-react';
import { useTranslations } from 'next-intl';
import { useCallback, useState } from 'react';

type Props = {
  onSuccess: () => void;
};

export const TextImport = ({ onSuccess }: Props) => {
  const { collection } = useCollectionContext();
  const t = useTranslations('page_documents');
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [isUploading, setIsUploading] = useState(false);

  const handleAdd = useCallback(async () => {
    if (!collection.id || !content.trim()) return;

    setIsUploading(true);
    try {
      const filename = title.trim()
        ? `${title.trim().slice(0, 200)}.txt`
        : `note-${Date.now()}.txt`;

      // Create a File object from the text — reuses the existing upload endpoint entirely
      const file = new File([content], filename, { type: 'text/plain' });

      await apiClient.defaultApi.collectionsCollectionIdDocumentsUploadPost({
        collectionId: collection.id,
        file,
      });

      onSuccess();
    } finally {
      setIsUploading(false);
    }
  }, [collection.id, title, content, onSuccess]);

  return (
    <div className="flex flex-col gap-4">
      <p className="text-muted-foreground text-sm">{t('import_text_desc')}</p>

      <div className="flex flex-col gap-1.5">
        <Label htmlFor="text-import-title">{t('import_text_title_label')}</Label>
        <Input
          id="text-import-title"
          placeholder={t('import_text_title_placeholder')}
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          disabled={isUploading}
          maxLength={200}
        />
      </div>

      <div className="flex flex-col gap-1.5">
        <Label htmlFor="text-import-content">{t('import_text_content_label')}</Label>
        <Textarea
          id="text-import-content"
          placeholder={t('import_text_content_placeholder')}
          value={content}
          onChange={(e) => setContent(e.target.value)}
          rows={10}
          disabled={isUploading}
          className="min-h-40 resize-y text-sm"
        />
      </div>

      <div className="flex justify-end">
        <Button onClick={handleAdd} disabled={!content.trim() || isUploading} className="min-w-24">
          {isUploading ? (
            <>
              <LoaderCircle className="animate-spin" />
              {t('import_text_uploading')}
            </>
          ) : (
            <>
              <Type />
              {t('import_text_btn')}
            </>
          )}
        </Button>
      </div>
    </div>
  );
};
