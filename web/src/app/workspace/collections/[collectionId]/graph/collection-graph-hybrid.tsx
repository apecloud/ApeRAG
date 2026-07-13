'use client';
/**
 * CollectionGraphHybrid
 *
 * Thin wrapper around CollectionGraph that sets the minimum 720px height so
 * shorter screens can scroll instead of compressing the canvas, and explicitly
 * routes data fetches through the marketplace-safe endpoints when
 * `marketplace={true}`.
 *
 * Data-source routing is already handled inside CollectionGraph via its
 * `marketplace` prop; this component adds the layout constraint and provides
 * the canonical import point for the hybrid view used by marketplace pages.
 */

import { CollectionGraph } from './collection-graph';

export const CollectionGraphHybrid = ({
  marketplace = false,
}: {
  marketplace?: boolean;
}) => {
  return (
    <div className="min-h-[720px] flex flex-col flex-1">
      <CollectionGraph marketplace={marketplace} />
    </div>
  );
};
