'use client';

import { useCallback, useEffect, useState } from 'react';

import { Collection } from '@/api';
import { apiClient } from '@/lib/api/client';
import { createContext, useContext } from 'react';

type CollectionContextProps = {
  collection: Collection;

  loadCollection: () => void;
};

const CollectionContext = createContext<CollectionContextProps>({
  collection: {},
  loadCollection: () => {},
});

export const useCollectionContext = () => useContext(CollectionContext);

export const CollectionProvider = ({
  collection: initCollection,
  children,
}: {
  children?: React.ReactNode;
  collection: Collection;
}) => {
  const [collection, setCollection] = useState<Collection>(initCollection);

  const loadCollection = useCallback(async () => {
    if (!collection?.id) {
      return;
    }
    const res = await apiClient.defaultApi.collectionsCollectionIdGet({
      collectionId: collection.id,
    });
    setCollection(res.data);
  }, [collection?.id]);

  useEffect(() => {
    setCollection(initCollection);
  }, [initCollection]);

  return (
    <CollectionContext.Provider value={{ collection, loadCollection }}>
      {children}
    </CollectionContext.Provider>
  );
};
