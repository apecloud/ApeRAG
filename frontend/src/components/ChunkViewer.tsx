import { DocumentPreview } from '@/api';
import { getAuthorizationHeader } from '@/models/user';
import { api } from '@/services';
import { ApeDocument, Chunk } from '@/types';
import { useDebounceFn, useRequest } from 'ahooks';
import { Col, Empty, List, Row, Segmented, Spin, Typography } from 'antd';
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

export const ChunkViewer = ({
  document: initialDoc,
  collectionId,
}: ChunkViewerProps) => {
  const [viewMode, setViewMode] = useState<
    'markdown' | 'pdf' | 'unsupported' | 'determining'
  >('determining');
  const [previewData, setPreviewData] = useState<DocumentPreview | null>(null);
  const [adaptedChunks, setAdaptedChunks] = useState<Chunk[]>([]);
  const [highlightedChunk, setHighlightedChunk] = useState<Chunk | null>(null);
  const [scrolledToChunkId, setScrolledToChunkId] = useState<string | null>(
    null,
  );
  const [numPages, setNumPages] = useState<number>(0);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
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
          if (data && data.chunks) {
            const chunks = data.chunks.map((chunk) => ({
              id: chunk.id || '',
              text: chunk.text || '',
              metadata: chunk.metadata || {},
            }));
            setAdaptedChunks(chunks);
          } else {
            setAdaptedChunks([]);
          }
        }
      },
    },
  );

  const canShowPdfPreview = useMemo(() => {
    if (!previewData) return false;
    const hasPdfSourceMap = adaptedChunks.some(
      (c) => c.metadata?.pdf_source_map,
    );
    if (!hasPdfSourceMap) return false;

    const isPdfFilename = previewData.doc_filename
      ?.toLowerCase()
      .endsWith('.pdf');
    return (
      !!previewData.converted_pdf_object_path ||
      (isPdfFilename && !!previewData.doc_object_path)
    );
  }, [previewData, adaptedChunks]);

  // Determine the best initial view mode once preview data is loaded
  useEffect(() => {
    if (previewData) {
      if (canShowPdfPreview) {
        setViewMode('pdf');
      } else if (previewData.markdown_content) {
        setViewMode('markdown');
      } else {
        setViewMode('unsupported');
      }
    }
  }, [previewData, canShowPdfPreview]);

  // Determine the PDF URL when view mode is switched to PDF
  useEffect(() => {
    if (viewMode === 'pdf' && canShowPdfPreview && previewData) {
      const pdfPath =
        previewData.converted_pdf_object_path || previewData.doc_object_path;
      if (pdfPath) {
        const url = `/api/v1/collections/${collectionId}/documents/${initialDoc.id!}/object?path=${encodeURIComponent(pdfPath)}`;
        setPdfUrl(url);
      }
    }
  }, [viewMode, canShowPdfPreview, previewData, collectionId, initialDoc.id]);

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

      if (viewMode === 'markdown' && highlightedChunk.metadata?.md_source_map) {
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
        highlightedChunk.metadata?.pdf_source_map
      ) {
        const sourceMaps = highlightedChunk.metadata.pdf_source_map;
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
      let focusChunkId = null;

      for (const chunk of adaptedChunks) {
        let top = Infinity;

        if (viewMode === 'pdf' && chunk.metadata?.pdf_source_map) {
          chunk.metadata.pdf_source_map.forEach((sourceMap: any) => {
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
        } else if (viewMode === 'markdown' && chunk.metadata?.md_source_map) {
          const [start_line] = chunk.metadata.md_source_map;
          const el = document.getElementById(`line-${start_line + 1}`);
          if (el) {
            top = el.offsetTop;
          }
        }

        if (top === Infinity) continue;

        if (top > containerScrollTop) {
          focusChunkId = chunk.id;
          break;
        }
      }

      if (
        container.scrollTop + container.clientHeight >=
        container.scrollHeight - 2
      ) {
        if (adaptedChunks.length > 0) {
          focusChunkId = adaptedChunks[adaptedChunks.length - 1].id;
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
      !highlightedChunk.metadata?.pdf_source_map ||
      viewMode !== 'pdf'
    ) {
      return null;
    }

    const pageDim = pageDimensions.get(pageNumber);
    if (!pageDim) {
      return null;
    }

    const highlights = [];
    for (const sourceMap of highlightedChunk.metadata.pdf_source_map) {
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
          whiteSpace: 'pre-wrap',
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

  const renderPdfView = () => {
    if (!pdfUrl) {
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
          file={pdfUrl}
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
    switch (viewMode) {
      case 'markdown':
        return <div key="markdown-view">{renderMarkdownView()}</div>;
      case 'pdf':
        return <div key="pdf-view">{renderPdfView()}</div>;
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
      {canShowPdfPreview && (
        <Segmented
          options={[
            { label: 'PDF', value: 'pdf' },
            { label: 'Markdown', value: 'markdown' },
          ]}
          value={viewMode}
          onChange={(value) => setViewMode(value as 'markdown' | 'pdf')}
          style={{ marginBottom: 16 }}
        />
      )}
      <Row gutter={16}>
        <Col span={12}>{renderContent()}</Col>
        <Col span={12}>
          <div
            ref={chunkListRef}
            onMouseLeave={() => setHighlightedChunk(null)}
            // Detect user-initiated scrolling on the right pane.
            // This sets a flag to prevent the left pane from syncing,
            // which avoids conflicts and improves the user's scrolling experience.
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
            <List
              header={
                <Typography.Title level={5}>
                  <FormattedMessage
                    id="chunk.viewer.chunks.title"
                    defaultMessage="Chunks"
                  />
                </Typography.Title>
              }
              bordered
              dataSource={adaptedChunks}
              renderItem={(item: Chunk) => {
                const isHighlighted = highlightedChunk?.id === item.id;
                return (
                  <List.Item
                    ref={(el) => {
                      if (el) chunkItemRefs.current.set(item.id, el);
                      else chunkItemRefs.current.delete(item.id);
                    }}
                    onMouseEnter={() => {
                      // The check for `isRightPaneScrolling` was removed here.
                      // While it prevented loops, it also caused a regression where highlighting
                      // wouldn't trigger during user scrolling. The new `onWheel` logic on the
                      // parent container is a more robust solution that handles this correctly.
                      setHighlightedChunk(item);
                    }}
                    className={styles.chunkListItem}
                    style={{
                      cursor: 'pointer',
                      backgroundColor: isHighlighted
                        ? '#e6f7ff'
                        : 'transparent',
                      transition: 'background-color 0.3s ease',
                    }}
                  >
                    <div className={styles.markdownContainer}>
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
                        {item.text}
                      </ReactMarkdown>
                    </div>
                  </List.Item>
                );
              }}
            />
          </div>
        </Col>
      </Row>
    </div>
  );
};
