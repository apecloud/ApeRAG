import { Collection } from '@/api';
import { DOCUMENT_DEFAULT_CONFIG } from '@/constants';
import { api } from '@/services';
import _ from 'lodash';
import { useCallback, useState } from 'react';
import { useModel } from 'umi';

export default () => {
  const { setLoading } = useModel('global');
  const [collection, setCollection] = useState<Collection>();
  const [collectionLoading, setCollectionLoading] = useState<boolean>(false);

  const [collections, setCollections] = useState<Collection[]>();
  const [collectionsLoading, setCollectionsLoading] = useState<boolean>(false);

  // collections
  const getCollections = useCallback(async () => {
    setLoading(true);
    setCollectionsLoading(true);
    const res = await api.collectionsGet();
    setLoading(false);
    setCollectionsLoading(false);
    setCollections(res.data.items);
  }, []);

  // collection
  const getCollection = useCallback(async (collectionId: string) => {
    setLoading(true);
    setCollectionLoading(true);
    const res = await api.collectionsCollectionIdGet({ collectionId });
    setCollection(res.data);
    setLoading(false);
    setCollectionLoading(false);
  }, []);

  const deleteCollection = useCallback(async (): Promise<
    boolean | undefined
  > => {
    if (!collection?.id) return;
    const res = await api.collectionsCollectionIdDelete({
      collectionId: collection.id,
    });
    return res.status === 200;
  }, [collection]);

  const createCollection = useCallback(
    async (values: Collection): Promise<string | undefined> => {
      const data = _.merge(
        {
          type: 'document',
          config: DOCUMENT_DEFAULT_CONFIG,
        },
        values,
      );
      setLoading(true);
      const res = await api.collectionsPost({
        collectionCreate: { ...data, config: data.config },
      });
      setLoading(false);
      return res.data.id;
    },
    [],
  );

  const updateCollection = useCallback(
    async (values: Collection): Promise<boolean | undefined> => {
      if (!collection?.id) return;
      const data = _.merge(collection, values);
      setLoading(true);
      const res = await api.collectionsCollectionIdPut({
        collectionId: collection.id,
        collectionUpdate: { ...data, config: data.config },
      });
      setLoading(false);
      const updated = res.status === 200;
      if (updated) getCollection(collection.id);
      return updated;
    },
    [collection],
  );

  return {
    collections,
    collectionsLoading,
    getCollections,
    setCollections,

    collection,
    collectionLoading,
    getCollection,
    setCollection,
    createCollection,
    updateCollection,
    deleteCollection,
  };
};
