import { api } from '@/services';
import { useRequest } from 'ahooks';
import { Image, Spin } from 'antd';
import { useEffect, useRef, useState } from 'react';

interface AuthAssetImageProps {
  src: string;
  collectionId: string;
  documentId: string;
  onLoad?: () => void;
}

export const AuthAssetImage = ({
  src,
  collectionId,
  documentId,
  onLoad,
}: AuthAssetImageProps) => {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [isIntersecting, setIntersecting] = useState(false);
  const placeholderRef = useRef<HTMLDivElement>(null);

  const { run: fetchImage, loading } = useRequest(
    async (assetPath: string) => {
      return api.getDocumentObject(
        {
          collectionId,
          documentId,
          path: assetPath,
        },
        {
          responseType: 'blob',
        },
      );
    },
    {
      manual: true,
      // Cache requests to prevent fetching the same image multiple times
      cacheKey: `auth-image-${src}`,
      onSuccess: (response: any) => {
        if (response && response.data) {
          const url = URL.createObjectURL(response.data);
          setImageUrl(url);
        }
      },
      onError: (error) => {
        console.error('Failed to fetch image:', error);
      },
    },
  );

  // Set up the Intersection Observer to lazy-load images.
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        // When the placeholder is in view, trigger the fetch.
        if (entry.isIntersecting) {
          setIntersecting(true);
          // We can disconnect the observer after the first intersection.
          if (placeholderRef.current) {
            observer.unobserve(placeholderRef.current);
          }
        }
      },
      {
        // Load images 100px before they enter the viewport.
        rootMargin: '100px',
      },
    );

    if (placeholderRef.current) {
      observer.observe(placeholderRef.current);
    }

    return () => {
      if (placeholderRef.current) {
        // eslint-disable-next-line react-hooks/exhaustive-deps
        observer.unobserve(placeholderRef.current);
      }
    };
  }, []);

  // Fetch the image only when it becomes visible.
  useEffect(() => {
    if (isIntersecting) {
      if (src && src.startsWith('asset://')) {
        const assetId = src.substring('asset://'.length).split('?')[0];
        fetchImage('assets/' + assetId);
      }
    }
  }, [isIntersecting, src, fetchImage]);

  // Effect to handle the lifecycle of the blob URL.
  useEffect(() => {
    return () => {
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  }, [imageUrl]);

  // If the image has been loaded, display it.
  if (imageUrl) {
    return (
      <Image
        src={imageUrl}
        alt={src}
        style={{ maxWidth: '100%' }}
        onLoad={onLoad}
      />
    );
  }

  // Otherwise, render a placeholder that will be observed for lazy loading.
  // A min-height is set to prevent layout shift when the image loads.
  return (
    <div
      ref={placeholderRef}
      style={{
        minHeight: 50,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: '100%',
      }}
    >
      {isIntersecting && loading && <Spin />}
    </div>
  );
};
