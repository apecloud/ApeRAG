'use client';

import { browserApiClient } from '@/lib/api/typed/browser';

export async function subscribeMarketplaceCollection(collectionId: string) {
  const { data } = await browserApiClient.POST(
    '/api/v1/marketplace/collections/{collection_id}/subscribe',
    {
      params: {
        path: { collection_id: collectionId },
      },
    },
  );
  return data;
}

export async function unsubscribeMarketplaceCollection(collectionId: string) {
  await browserApiClient.DELETE(
    '/api/v1/marketplace/collections/{collection_id}/subscribe',
    {
      params: {
        path: { collection_id: collectionId },
      },
    },
  );
}
