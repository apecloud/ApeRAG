import { DocumentPreview, VisionChunk } from '@/api';
import { getAuthorizationHeader } from '@/models/user';
import { api } from '@/services';
import { ApeDocument, Chunk } from '@/types';
import { useDebounceFn, useRequest } from 'ahooks';
import { Col, Empty, List, Row, Segmented, Spin } from 'antd';
import 'katex/dist/katex.min.css'; // Import katex css
import { useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown, { Components } from 'react-markdown';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { FormattedMessage } from 'umi';
import { visit } from 'unist-util-visit';
import { AuthAssetImage } from './AuthAssetImage';
import styles from './ChunkViewer.module.css';

// Set up worker
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString();

interface ChunkViewerProps {
  document: ApeDocument;
  collectionId: string;
}

// A rehype plugin to add line number IDs to block-level elements
const rehypeAddLineIds = () => {
  return (tree: any) => {
    visit(tree, 'element', (node) => {
      if (node.position) {
        const startLine = node.position.start.line;
        node.properties = node.properties || {};
        node.properties.id = `line-${startLine}`;
      }
    });
  };
};

const processLongText = (text: string) => {
  // Manually insert zero-width spaces to force wrapping and prevent excessively long text (e.g., thousands of '.' in a row),
  // which can cause the browser's UI thread to hang during rendering.
  // TODO: avoid inserting characters into an inlined image (e.g., "![](data:image/png;base64,xxxxxx)")
  const processedText = (text || '').replace(/\S{400,}/g, (word: string) => {
    return word.replace(/(.{400})/g, '$1\u200B'); // Insert zero-width space every 400 characters
  });
  return processedText;
};

export const ChunkViewer = ({
  document: initialDoc,
  collectionId,
}: ChunkViewerProps) => {
  const [viewMode, setViewMode] = useState<
    'markdown' | 'pdf' | 'image' | 'unsupported' | 'determining'
  >('determining');
  const [rightViewMode, setRightViewMode] = useState<'chunks' | 'vision'>(
    'chunks',
  );
  const [rightPaneOptions, setRightPaneOptions] = useState<
    { label: string; value: 'chunks' | 'vision' }[]
  >([]);
  const [previewData, setPreviewData] = useState<DocumentPreview | null>(null);
  const [adaptedChunks, setAdaptedChunks] = useState<Chunk[]>([]);
  const [visionChunks, setVisionChunks] = useState<VisionChunk[]>([]);
  const [highlightedChunk, setHighlightedChunk] = useState<
    Chunk | VisionChunk | null
  >(null);
  const [scrolledToChunkId, setScrolledToChunkId] = useState<string | null>(
    null,
  );
  const [numPages, setNumPages] = useState<number>(0);
  const [leftPaneOptions, setLeftPaneOptions] = useState<
    { label: string; value: 'pdf' | 'markdown' | 'image' }[]
  >([]);
  const [leftPaneContentUrl, setLeftPaneContentUrl] = useState<string | null>(
    null,
  );
  const pdfContainerRef = useRef<HTMLDivElement>(null);
  const markdownContainerRef = useRef<HTMLDivElement>(null);
  const chunkListRef = useRef<HTMLDivElement>(null);
  const pageRefs = useRef<Map<number, HTMLElement>>(new Map());
  const chunkItemRefs = useRef<Map<string, HTMLElement>>(new Map());
  const [pdfWidth, setPdfWidth] = useState<number | undefined>();
  const lastHighlightedLines = useRef<number[]>([]);
  // Refs to track programmatic scrolling states to prevent infinite loops.
  // When a pane is scrolled programmatically, its corresponding flag is set to true.
  // This prevents the scroll event from triggering a reciprocal scroll in the other pane.
  // A timeout resets the flag after the scroll animation, re-enabling user-triggered sync.
  const isLeftPaneScrolling = useRef(false);
  const leftPaneScrollTimeout = useRef<NodeJS.Timeout | null>(null);
  const isRightPaneScrolling = useRef(false);
  const rightPaneScrollTimeout = useRef<NodeJS.Timeout | null>(null);
  // This flag tracks if the user is actively scrolling the right pane (chunk list).
  // It's used to temporarily disable the left pane's scroll-to-sync behavior
  // to prevent jittery interactions when the user is in control of the right pane.
  const isUserScrollingRight = useRef(false);
  const userScrollTimeout = useRef<NodeJS.Timeout | null>(null);

  // Fetch all preview data in one go
  const { loading: previewLoading, error: previewError } = useRequest(
    () => {
      if (!initialDoc.id || !collectionId) return Promise.resolve(null);
      return api.getDocumentPreview({
        collectionId,
        documentId: initialDoc.id,
      });
    },
    {
      ready: !!initialDoc.id && !!collectionId,
      onSuccess: (response) => {
        if (response) {
          const data = response.data as DocumentPreview | null;
          setPreviewData(data);
          if (data) {
            setAdaptedChunks(
              data.chunks?.map((chunk) => ({
                id: chunk.id || '',
                text: chunk.text || '',
                metadata: chunk.metadata || {},
              })) || [],
            );
            setVisionChunks(
              data.vision_chunks?.map((chunk) => ({
                id: chunk.id || '',
                asset_id: chunk.asset_id || '',
                text: chunk.text || '',
                metadata: chunk.metadata || {},
              })) || [],
            );
          } else {
            setAdaptedChunks([]);
            setVisionChunks([]);
          }
        }
      },
    },
  );

  const isOriginalImage = useMemo(() => {
    const filename = previewData?.doc_filename?.toLowerCase();
    if (!filename) return false;
    return [
      '.jpg',
      '.jpeg',
      '.png',
      '.bmp',
      '.gif',
      '.webp',
      '.tiff',
      '.tif',
    ].some((ext) => filename.endsWith(ext));
  }, [previewData?.doc_filename]);

  const canShowPdfPreview = useMemo(() => {
    if (!previewData) return false;
    return !!previewData.converted_pdf_object_path;
  }, [previewData]);

  // Determine the best initial view mode once preview data is loaded
  useEffect(() => {
    if (previewData) {
      const options: { label: string; value: 'pdf' | 'markdown' | 'image' }[] =
        [];
      if (isOriginalImage) {
        options.push({ label: 'Image', value: 'image' });
      } else if (canShowPdfPreview) {
        // Only show PDF option if it's not an original image
        options.push({ label: 'PDF', value: 'pdf' });
      }
      if (previewData.markdown_content) {
        options.push({ label: 'Markdown', value: 'markdown' });
      }

      setLeftPaneOptions(options);

      if (options.length > 0) {
        const priority = ['image', 'pdf', 'markdown'];
        for (const p of priority) {
          if (options.some((o) => o.value === p)) {
            setViewMode(p as 'image' | 'pdf' | 'markdown');
            break;
          }
        }
      } else {
        setViewMode('unsupported');
      }

      // Determine right pane options and default view
      const rightOptions: { label: string; value: 'chunks' | 'vision' }[] = [];
      const hasChunks = adaptedChunks.length > 0;
      const hasVisionChunks = visionChunks.length > 0;

      if (hasChunks) {
        rightOptions.push({ label: 'Chunks', value: 'chunks' });
      }
      if (hasVisionChunks) {
        rightOptions.push({ label: 'Visual Descriptions', value: 'vision' });
      }
      setRightPaneOptions(rightOptions);

      if (hasVisionChunks && !hasChunks) {
        setRightViewMode('vision');
      } else {
        setRightViewMode('chunks');
      }
    }
  }, [
    previewData,
    canShowPdfPreview,
    isOriginalImage,
    adaptedChunks.length,
    visionChunks.length,
  ]);

  // Determine the PDF or Image URL when view mode is switched
  useEffect(() => {
    if (!previewData || !initialDoc.id) return;

    let objectPath: string | undefined;
    if (viewMode === 'pdf' && canShowPdfPreview) {
      objectPath = previewData.converted_pdf_object_path;
    } else if (viewMode === 'image' && isOriginalImage) {
      objectPath = previewData.doc_object_path;
    }

    if (objectPath) {
      const url = `/api/v1/collections/${collectionId}/documents/${initialDoc.id}/object?path=${encodeURIComponent(objectPath)}`;
      setLeftPaneContentUrl(url);
    }
  }, [
    viewMode,
    canShowPdfPreview,
    isOriginalImage,
    previewData,
    collectionId,
    initialDoc.id,
  ]);

  const pdfOptions = useMemo(() => {
    return {
      httpHeaders: getAuthorizationHeader(),
      rangeChunkSize: 256 * 1024,
      disableStream: true,
      disableAutoFetch: true,
    };
  }, []);

  const [pageDimensions, setPageDimensions] = useState(new Map());

  const { run: syncAndHighlight } = useDebounceFn(
    () => {
      // Clear previous highlights
      lastHighlightedLines.current.forEach((lineNum) => {
        const el = document.getElementById(`line-${lineNum}`);
        if (el) {
          el.classList.remove(styles.highlightedLine);
        }
      });
      lastHighlightedLines.current = [];

      if (!highlightedChunk) return;

      const scrollIfNeeded = (
        containerElement: HTMLElement,
        overallTop: number,
        overallBottom: number,
      ) => {
        const containerScrollTop = containerElement.scrollTop;
        const containerHeight = containerElement.clientHeight;
        const overallHeight = overallBottom - overallTop;

        const isFullyVisible =
          overallTop >= containerScrollTop &&
          overallBottom <= containerScrollTop + containerHeight;

        if (!isFullyVisible) {
          let newScrollTop;
          // If the chunk is taller than the viewport, align its top with the viewport's top.
          // Otherwise, center the chunk in the viewport.
          if (overallHeight > containerHeight) {
            newScrollTop = overallTop;
          } else {
            newScrollTop = overallTop - (containerHeight - overallHeight) / 2;
          }

          // Programmatically scroll the left pane (document view)
          isLeftPaneScrolling.current = true;
          if (leftPaneScrollTimeout.current) {
            clearTimeout(leftPaneScrollTimeout.current);
          }

          containerElement.scrollTo({
            top: newScrollTop,
            behavior: 'smooth',
          });

          leftPaneScrollTimeout.current = setTimeout(() => {
            isLeftPaneScrolling.current = false;
          }, 500); // Should be longer than the smooth scroll duration
        }
      };

      if (
        viewMode === 'markdown' &&
        highlightedChunk.metadata &&
        'md_source_map' in highlightedChunk.metadata &&
        highlightedChunk.metadata.md_source_map
      ) {
        const [start_line, end_line] = highlightedChunk.metadata.md_source_map;
        const linesToHighlight = Array.from(
          { length: end_line - start_line + 1 },
          (_, i) => start_line + 1 + i,
        );

        linesToHighlight.forEach((lineNum) => {
          const el = document.getElementById(`line-${lineNum}`);
          if (el) {
            el.classList.add(styles.highlightedLine);
          }
        });
        lastHighlightedLines.current = linesToHighlight;

        const startElement = document.getElementById(`line-${start_line + 1}`);
        const endElement =
          document.getElementById(`line-${end_line + 1}`) || startElement;
        const containerElement = markdownContainerRef.current;

        if (startElement && endElement && containerElement) {
          const overallTop = startElement.offsetTop;
          const overallBottom = endElement.offsetTop + endElement.offsetHeight;
          scrollIfNeeded(containerElement, overallTop, overallBottom);
        }
      } else if (
        viewMode === 'pdf' &&
        highlightedChunk.metadata &&
        'page_idx' in highlightedChunk.metadata &&
        !('pdf_source_map' in highlightedChunk.metadata)
      ) {
        const pageNumber = (highlightedChunk.metadata.page_idx as number) + 1;
        const pageElement = pageRefs.current.get(pageNumber);
        const containerElement = pdfContainerRef.current;

        if (pageElement && containerElement) {
          scrollIfNeeded(
            containerElement,
            pageElement.offsetTop,
            pageElement.offsetTop + pageElement.offsetHeight,
          );
        }
      } else if (
        viewMode === 'pdf' &&
        highlightedChunk.metadata &&
        'pdf_source_map' in highlightedChunk.metadata &&
        highlightedChunk.metadata.pdf_source_map
      ) {
        const sourceMaps = highlightedChunk.metadata.pdf_source_map as any[];
        if (!sourceMaps || sourceMaps.length === 0) return;

        const containerElement = pdfContainerRef.current;
        if (!containerElement) return;

        let overallTop = Infinity;
        let overallBottom = -Infinity;

        for (const sourceMap of sourceMaps) {
          const pageNumber = sourceMap.page_idx + 1;
          const pageElement = pageRefs.current.get(pageNumber);
          const pageDim = pageDimensions.get(pageNumber);

          if (pageElement && pageDim) {
            const [, y1, , y2] = sourceMap.bbox;
            const scale = (pdfWidth || pageDim.width) / pageDim.width;
            const pageTopInContainer = pageElement.offsetTop;
            const bboxTop = pageTopInContainer + y1 * scale;
            const bboxBottom = pageTopInContainer + y2 * scale;
            if (bboxTop < overallTop) overallTop = bboxTop;
            if (bboxBottom > overallBottom) overallBottom = bboxBottom;
          }
        }

        if (overallTop === Infinity) return;

        scrollIfNeeded(containerElement, overallTop, overallBottom);
      }
    },
    { wait: 100 },
  );

  useEffect(() => {
    syncAndHighlight();
  }, [highlightedChunk, viewMode, syncAndHighlight]);

  // Use ResizeObserver to re-run sync function when layout changes (e.g., images load)
  useEffect(() => {
    const container = markdownContainerRef.current;
    if (!container) return;

    const observer = new ResizeObserver(() => {
      syncAndHighlight();
    });

    // We observe the direct child rendered by ReactMarkdown
    if (container.firstChild && container.firstChild instanceof Element) {
      observer.observe(container.firstChild);
    }

    return () => {
      observer.disconnect();
    };
  }, [previewData, syncAndHighlight]);

  const { run: handleScroll } = useDebounceFn(
    () => {
      // Ignore scroll events under two conditions:
      // 1. The left pane is being scrolled programmatically (by highlighting a chunk).
      // 2. The user is actively scrolling the right pane (via mouse wheel).
      // This prevents jitter and conflicting scroll behaviors.
      if (isLeftPaneScrolling.current || isUserScrollingRight.current) return;

      const container =
        viewMode === 'pdf'
          ? pdfContainerRef.current
          : markdownContainerRef.current;
      if (!container) return;

      const containerScrollTop = container.scrollTop;
      let focusChunkId: string | null = null;

      const currentDataSource =
        rightViewMode === 'chunks' ? adaptedChunks : visionChunks;

      for (const chunk of currentDataSource) {
        let top = Infinity;

        if (
          viewMode === 'pdf' &&
          chunk.metadata &&
          'page_idx' in chunk.metadata &&
          !('pdf_source_map' in chunk.metadata)
        ) {
          const pageNumber = (chunk.metadata.page_idx as number) + 1;
          const pageElement = pageRefs.current.get(pageNumber);
          if (pageElement) {
            top = pageElement.offsetTop;
          }
        } else if (
          viewMode === 'pdf' &&
          chunk.metadata &&
          'pdf_source_map' in chunk.metadata &&
          chunk.metadata.pdf_source_map
        ) {
          (chunk.metadata.pdf_source_map as any[]).forEach((sourceMap: any) => {
            const pageNumber = sourceMap.page_idx + 1;
            const pageElement = pageRefs.current.get(pageNumber);
            const pageDim = pageDimensions.get(pageNumber);
            if (pageElement && pageDim) {
              const [, y1] = sourceMap.bbox;
              const scale = (pdfWidth || pageDim.width) / pageDim.width;
              const pageTopInContainer = pageElement.offsetTop;
              const bboxTop = pageTopInContainer + y1 * scale;
              if (bboxTop < top) top = bboxTop;
            }
          });
        } else if (
          viewMode === 'markdown' &&
          chunk.metadata &&
          'md_source_map' in chunk.metadata &&
          chunk.metadata.md_source_map
        ) {
          const [start_line] = chunk.metadata.md_source_map;
          const el = document.getElementById(`line-${start_line + 1}`);
          if (el) {
            top = el.offsetTop;
          }
        }

        if (top === Infinity) continue;

        if (top > containerScrollTop) {
          focusChunkId = chunk.id || null;
          break;
        }
      }

      if (
        container.scrollTop + container.clientHeight >=
        container.scrollHeight - 2
      ) {
        if (currentDataSource.length > 0) {
          focusChunkId =
            currentDataSource[currentDataSource.length - 1].id || null;
        }
      }

      if (focusChunkId && focusChunkId !== scrolledToChunkId) {
        setScrolledToChunkId(focusChunkId);
      }
    },
    { wait: 150 },
  );

  useEffect(() => {
    if (scrolledToChunkId) {
      const chunkItem = chunkItemRefs.current.get(scrolledToChunkId);
      if (chunkItem) {
        // Programmatically scroll the right pane (chunk list)
        isRightPaneScrolling.current = true;
        if (rightPaneScrollTimeout.current) {
          clearTimeout(rightPaneScrollTimeout.current);
        }

        chunkItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reset the flag after the scroll animation
        rightPaneScrollTimeout.current = setTimeout(() => {
          isRightPaneScrolling.current = false;
          // Manually trigger a scroll check to update highlighting if needed
          handleScroll();
        }, 500); // Should be longer than the smooth scroll duration
      }
    }
  }, [scrolledToChunkId, handleScroll]);

  useEffect(() => {
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        setPdfWidth(entry.contentRect.width);
      }
    });
    const container = pdfContainerRef.current;
    if (container) {
      observer.observe(container);
    }
    return () => {
      if (container) {
        observer.unobserve(container);
      }
    };
  }, []);

  const onDocumentLoadSuccess = ({ numPages }: { numPages: number }) => {
    setNumPages(numPages);
  };

  const onPageRenderSuccess = (page: any) => {
    setPageDimensions((prev) =>
      new Map(prev).set(page.pageNumber, {
        width: page.width,
        height: page.height,
      }),
    );
  };

  const renderPdfHighlight = (pageNumber: number) => {
    if (
      !highlightedChunk ||
      !highlightedChunk.metadata ||
      !('pdf_source_map' in highlightedChunk.metadata) ||
      !highlightedChunk.metadata.pdf_source_map ||
      viewMode !== 'pdf'
    ) {
      return null;
    }

    const pageDim = pageDimensions.get(pageNumber);
    if (!pageDim) {
      return null;
    }

    const highlights = [];
    for (const sourceMap of highlightedChunk.metadata.pdf_source_map as any[]) {
      if (sourceMap.page_idx === pageNumber - 1) {
        const [x1, y1, x2, y2] = sourceMap.bbox;
        const { width, height } = pageDim;
        const style: React.CSSProperties = {
          position: 'absolute',
          left: `${(x1 / width) * 100}%`,
          top: `${(y1 / height) * 100}%`,
          width: `${((x2 - x1) / width) * 100}%`,
          height: `${((y2 - y1) / height) * 100}%`,
          backgroundColor: 'rgba(255, 255, 0, 0.3)',
          pointerEvents: 'none',
          zIndex: 1,
        };
        highlights.push(
          <div key={`${sourceMap.page_idx}-${x1}-${y1}`} style={style} />,
        );
      }
    }
    return highlights;
  };

  const markdownComponents: Components = useMemo(
    () => ({
      p: (props) => {
        const { node } = props;
        if (
          node &&
          node.children[0]?.type === 'element' &&
          node.children[0]?.tagName === 'img'
        ) {
          return <>{props.children}</>;
        }
        return <p {...props} />;
      },
      img: (props) => {
        const { src } = props;
        if (src && src.startsWith('asset://')) {
          return (
            <AuthAssetImage
              src={src}
              collectionId={collectionId}
              documentId={initialDoc.id!}
              onLoad={syncAndHighlight}
            />
          );
        }
        return <img {...props} onLoad={syncAndHighlight} />;
      },
    }),
    [collectionId, initialDoc.id, syncAndHighlight],
  );

  const renderMarkdownView = () => {
    if (!previewData?.markdown_content) {
      return (
        <Empty
          description={
            <FormattedMessage
              id="chunk.viewer.markdown.empty"
              defaultMessage="No markdown content available for preview."
            />
          }
        />
      );
    }

    return (
      <div
        ref={markdownContainerRef}
        onScroll={handleScroll}
        className={styles.markdownContainer}
        style={{
          position: 'relative',
          height: '80vh',
          overflowY: 'auto',
          border: '1px solid #f0f0f0',
          padding: '16px',
          overflowWrap: 'break-word',
          wordBreak: 'break-word',
        }}
      >
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[rehypeAddLineIds, rehypeRaw, rehypeKatex]}
          components={markdownComponents}
          urlTransform={(url) =>
            url.startsWith('asset://')
              ? url
              : new URL(url, window.location.href).href
          }
        >
          {previewData.markdown_content}
        </ReactMarkdown>
      </div>
    );
  };

  const renderImageView = () => {
    if (!leftPaneContentUrl) {
      return (
        <div
          style={{
            height: '80vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Spin />
        </div>
      );
    }
    return (
      <div
        ref={markdownContainerRef} // Using markdown ref for simplicity as it's also a container
        className={styles.markdownContainer}
        style={{
          position: 'relative',
          height: '80vh',
          overflowY: 'auto',
          border: '1px solid #f0f0f0',
          padding: '16px',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
        }}
      >
        <img
          src={leftPaneContentUrl}
          alt={previewData?.doc_filename || 'Document Image'}
          style={{ maxWidth: '100%', maxHeight: '100%' }}
        />
      </div>
    );
  };

  const renderPdfView = () => {
    if (!leftPaneContentUrl) {
      return (
        <div
          style={{
            height: '80vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Spin />
        </div>
      );
    }

    return (
      <div
        ref={pdfContainerRef}
        onScroll={handleScroll}
        style={{
          height: '80vh',
          overflowY: 'auto',
          backgroundColor: '#f0f0f0',
        }}
      >
        <Document
          file={leftPaneContentUrl}
          options={pdfOptions}
          onLoadSuccess={onDocumentLoadSuccess}
        >
          {Array.from(new Array(numPages), (el, index) => (
            <div
              key={`page_${index + 1}`}
              ref={(el) => {
                if (el) pageRefs.current.set(index + 1, el);
                else pageRefs.current.delete(index + 1);
              }}
              style={{
                display: 'flex',
                justifyContent: 'center',
                marginBottom: '8px',
              }}
            >
              <div style={{ position: 'relative' }}>
                <Page
                  pageNumber={index + 1}
                  width={pdfWidth}
                  onRenderSuccess={onPageRenderSuccess}
                />
                {renderPdfHighlight(index + 1)}
              </div>
            </div>
          ))}
        </Document>
      </div>
    );
  };

  if (previewLoading) {
    return <Spin />;
  }

  if (previewError || !previewData) {
    return (
      <Empty
        description={
          <FormattedMessage
            id="chunk.viewer.data.loadFailed"
            defaultMessage="Failed to load preview data."
          />
        }
      />
    );
  }

  const renderContent = () => {
    // Always render the container, even if empty, to maintain layout
    const hasContent =
      (viewMode === 'markdown' && !!previewData?.markdown_content) ||
      (viewMode === 'pdf' && canShowPdfPreview) ||
      (viewMode === 'image' && isOriginalImage);

    if (!hasContent && viewMode !== 'determining') {
      return (
        <div
          style={{
            height: '80vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            border: '1px solid #f0f0f0',
          }}
        >
          <Empty
            description={
              <FormattedMessage
                id="chunk.viewer.unsupported"
                defaultMessage="This document format is not supported for preview."
              />
            }
          />
        </div>
      );
    }

    switch (viewMode) {
      case 'markdown':
        return <div key="markdown-view">{renderMarkdownView()}</div>;
      case 'pdf':
        return <div key="pdf-view">{renderPdfView()}</div>;
      case 'image':
        return <div key="image-view">{renderImageView()}</div>;
      case 'determining':
        return (
          <div
            style={{
              height: '80vh',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Spin />
          </div>
        );
      default:
        return (
          <div
            style={{
              height: '80vh',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Empty
              description={
                <FormattedMessage
                  id="chunk.viewer.unsupported"
                  defaultMessage="This document format is not supported for preview."
                />
              }
            />
          </div>
        );
    }
  };

  return (
    <div>
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={12}>
          {leftPaneOptions.length > 0 && (
            <Segmented
              options={leftPaneOptions}
              value={viewMode}
              onChange={(value) =>
                setViewMode(value as 'markdown' | 'pdf' | 'image')
              }
            />
          )}
        </Col>
        <Col span={12}>
          {rightPaneOptions.length > 0 && (
            <Segmented
              options={rightPaneOptions}
              value={rightViewMode}
              onChange={(value) =>
                setRightViewMode(value as 'chunks' | 'vision')
              }
            />
          )}
        </Col>
      </Row>
      <Row gutter={16}>
        <Col span={12}>{renderContent()}</Col>
        <Col span={12}>
          <div
            ref={chunkListRef}
            onMouseLeave={() => setHighlightedChunk(null)}
            onWheel={() => {
              isUserScrollingRight.current = true;
              if (userScrollTimeout.current) {
                clearTimeout(userScrollTimeout.current);
              }
              userScrollTimeout.current = setTimeout(
                () => (isUserScrollingRight.current = false),
                200,
              );
            }}
            style={{ height: '80vh', overflowY: 'auto' }}
          >
            {adaptedChunks.length > 0 || visionChunks.length > 0 ? (
              <List
                bordered
                dataSource={
                  rightViewMode === 'chunks' ? adaptedChunks : visionChunks
                }
                renderItem={(item: Chunk | VisionChunk) => {
                  const isHighlighted = highlightedChunk?.id === item.id;
                  const pageIdx =
                    item.metadata && 'page_idx' in item.metadata
                      ? (item.metadata.page_idx as number)
                      : null;

                  return (
                    <List.Item
                      ref={(el) => {
                        if (el) chunkItemRefs.current.set(item.id || '', el);
                        else chunkItemRefs.current.delete(item.id || '');
                      }}
                      onMouseEnter={() => {
                        setHighlightedChunk(item);
                      }}
                      className={styles.chunkListItem}
                      style={{
                        cursor: 'pointer',
                        backgroundColor: isHighlighted
                          ? '#e6f7ff'
                          : 'transparent',
                        transition: 'background-color 0.3s ease',
                        display: 'block', // Allow block-level elements inside
                      }}
                    >
                      {rightViewMode === 'vision' && pageIdx !== null && (
                        <div
                          style={{
                            fontWeight: 'bold',
                            marginBottom: '8px',
                            paddingBottom: '8px',
                            borderBottom: '1px solid #f0f0f0',
                          }}
                        >
                          Page: {pageIdx + 1}
                        </div>
                      )}
                      <div
                        className={styles.markdownContainer}
                        style={{
                          overflowWrap: 'break-word',
                          wordBreak: 'break-word',
                        }}
                      >
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm, remarkMath]}
                          rehypePlugins={[rehypeRaw, rehypeKatex]}
                          components={markdownComponents}
                          urlTransform={(url) =>
                            url.startsWith('asset://')
                              ? url
                              : new URL(url, window.location.href).href
                          }
                        >
                          {processLongText(item.text || '')}
                        </ReactMarkdown>
                      </div>
                    </List.Item>
                  );
                }}
              />
            ) : (
              <div
                style={{
                  height: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: '1px solid #f0f0f0',
                }}
              >
                <Empty
                  description={
                    <FormattedMessage
                      id="chunk.viewer.chunks.empty"
                      defaultMessage="No chunks available for this document."
                    />
                  }
                />
              </div>
            )}
          </div>
        </Col>
      </Row>
    </div>
  );
};
