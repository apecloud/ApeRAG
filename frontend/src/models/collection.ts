import { Collection, CollectionView, CollectionViewList, SharingStatusResponse, SharedCollection, SharedCollectionList } from '@/api';
import { DOCUMENT_DEFAULT_CONFIG } from '@/constants';
import { api } from '@/services';
import _ from 'lodash';
import { useCallback, useState } from 'react';
import { useModel } from 'umi';

export default () => {
  const { setLoading } = useModel('global');
  const [collection, setCollection] = useState<Collection>();
  const [collectionLoading, setCollectionLoading] = useState<boolean>(false);

  const [collections, setCollections] = useState<CollectionView[]>();
  const [collectionsLoading, setCollectionsLoading] = useState<boolean>(false);

  // sharing status
  const [sharingStatus, setSharingStatus] = useState<SharingStatusResponse>();
  const [sharingLoading, setSharingLoading] = useState<boolean>(false);

  // marketplace collections
  const [marketplaceCollections, setMarketplaceCollections] = useState<SharedCollection[]>();
  const [marketplaceLoading, setMarketplaceLoading] = useState<boolean>(false);
  const [marketplacePagination, setMarketplacePagination] = useState({
    current: 1,
    pageSize: 12,
    total: 0,
  });

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

  // sharing effects
  const getSharingStatus = useCallback(async (collectionId: string) => {
    setSharingLoading(true);
    try {
      const res = await api.collectionsCollectionIdSharingGet({ collectionId });
      setSharingStatus(res.data);
      return res.data;
    } catch (error) {
      console.error('Failed to get sharing status:', error);
    } finally {
      setSharingLoading(false);
    }
  }, []);

  const publishCollection = useCallback(async (collectionId: string): Promise<boolean> => {
    setSharingLoading(true);
    try {
      const res = await api.collectionsCollectionIdSharingPost({ collectionId });
      const success = res.status === 204;
      if (success) {
        // Refresh sharing status
        await getSharingStatus(collectionId);
      }
      return success;
    } catch (error) {
      console.error('Failed to publish collection:', error);
      return false;
    } finally {
      setSharingLoading(false);
    }
  }, [getSharingStatus]);

  const unpublishCollection = useCallback(async (collectionId: string): Promise<boolean> => {
    setSharingLoading(true);
    try {
      const res = await api.collectionsCollectionIdSharingDelete({ collectionId });
      const success = res.status === 204;
      if (success) {
        // Refresh sharing status
        await getSharingStatus(collectionId);
      }
      return success;
    } catch (error) {
      console.error('Failed to unpublish collection:', error);
      return false;
    } finally {
      setSharingLoading(false);
    }
  }, [getSharingStatus]);

  // marketplace effects
  const getMarketplaceCollections = useCallback(async (page: number = 1, pageSize: number = 12) => {
    setMarketplaceLoading(true);
    try {
      const res = await api.marketplaceCollectionsGet({ page, pageSize });
      const data = res.data;
      setMarketplaceCollections(data.items || []);
      setMarketplacePagination({
        current: data.page || 1,
        pageSize: data.page_size || 12,
        total: data.total || 0,
      });
      return data;
    } catch (error) {
      console.error('Failed to get marketplace collections:', error);
      setMarketplaceCollections([]);
    } finally {
      setMarketplaceLoading(false);
    }
  }, []);

  const subscribeToCollection = useCallback(async (collectionId: string): Promise<boolean> => {
    try {
      const res = await api.marketplaceCollectionsCollectionIdSubscribePost({ collectionId });
      const success = res.status === 200;
      if (success) {
        // Refresh marketplace collections to update subscription status
        await getMarketplaceCollections(marketplacePagination.current, marketplacePagination.pageSize);
      }
      return success;
    } catch (error) {
      console.error('Failed to subscribe to collection:', error);
      return false;
    }
  }, [getMarketplaceCollections, marketplacePagination]);

  const unsubscribeFromCollection = useCallback(async (collectionId: string): Promise<boolean> => {
    try {
      const res = await api.marketplaceCollectionsCollectionIdSubscribeDelete({ collectionId });
      const success = res.status === 200;
      if (success) {
        // Refresh marketplace collections to update subscription status
        await getMarketplaceCollections(marketplacePagination.current, marketplacePagination.pageSize);
      }
      return success;
    } catch (error) {
      console.error('Failed to unsubscribe from collection:', error);
      return false;
    }
  }, [getMarketplaceCollections, marketplacePagination]);

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

    // sharing
    sharingStatus,
    sharingLoading,
    getSharingStatus,
    setSharingStatus,
    publishCollection,
    unpublishCollection,

    // marketplace
    marketplaceCollections,
    marketplaceLoading,
    marketplacePagination,
    getMarketplaceCollections,
    setMarketplaceCollections,
    setMarketplacePagination,
    subscribeToCollection,
    unsubscribeFromCollection,
  };
};
