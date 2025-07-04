import { getAuthorizationHeader } from '@/models/user';
import { ApeDocument, Chunk } from '@/types';
import { useDebounceFn, useRequest } from 'ahooks';
import { Col, Empty, List, Row, Segmented, Spin, Typography } from 'antd';
import axios from 'axios';
import { useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Document, Page, pdfjs } from 'react-pdf';
import { FormattedMessage } from 'umi';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';
import remarkGfm from 'remark-gfm';
import styles from './ChunkViewer.module.css';

// Set up worker
pdfjs.GlobalWorkerOptions.workerSrc = `https://cdn.jsdelivr.net/npm/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;
// pdfjs.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;

interface ChunkViewerProps {
  document: ApeDocument;
  collectionId: string;
}

export const ChunkViewer = ({ document: initialDoc, collectionId }: ChunkViewerProps) => {
  const [viewMode, setViewMode] = useState<'markdown' | 'pdf' | 'unsupported'>('markdown');
  const [previewData, setPreviewData] = useState<any>(null);
  const [pdfFile, setPdfFile] = useState<any>(null);
  const [highlightedChunk, setHighlightedChunk] = useState<Chunk | null>(null);
  const [scrolledToChunkId, setScrolledToChunkId] = useState<string | null>(null);
  const [numPages, setNumPages] = useState<number>(0);
  const pdfContainerRef = useRef<HTMLDivElement>(null);
  const markdownContainerRef = useRef<HTMLDivElement>(null);
  const chunkListRef = useRef<HTMLDivElement>(null);
  const lineRefs = useRef<Map<number, HTMLElement>>(new Map());
  const pageRefs = useRef<Map<number, HTMLElement>>(new Map());
  const chunkItemRefs = useRef<Map<string, HTMLElement>>(new Map());
  const isHovering = useRef(false);
  const [pdfWidth, setPdfWidth] = useState<number | undefined>();

  // Fetch all preview data in one go
  const { loading: previewLoading, error: previewError } = useRequest(
    async () => {
      if (!initialDoc.id || !collectionId) return null;
      const response = await axios.get(
        `/api/v1/collections/${collectionId}/documents/${initialDoc.id}/preview`,
        { headers: getAuthorizationHeader() },
      );
      return response.data;
    },
    {
      ready: !!initialDoc.id && !!collectionId,
      onSuccess: (data) => {
        setPreviewData(data);
      },
    },
  );

  // Fetch PDF blob on demand
  const { run: fetchPdf, loading: pdfLoading } = useRequest(
    async (path: string) => {
      const response = await axios.get(
        `/api/v1/collections/${collectionId}/documents/${initialDoc.id}/object`,
        {
          params: { path },
          headers: getAuthorizationHeader(),
          responseType: 'blob',
        },
      );
      return response.data;
    },
    {
      manual: true,
      onSuccess: (data) => {
        setPdfFile(data);
      },
    },
  );

  const canShowPdfPreview = useMemo(() => {
    if (!previewData) return false;
    const hasPdfSourceMap = previewData.chunks?.some((c: Chunk) => c.metadata?.pdf_source_map);
    if (!hasPdfSourceMap) return false;

    return !!(previewData.converted_pdf_object_path || (previewData.doc_filename?.toLowerCase().endsWith('.pdf') && previewData.doc_object_path));
  }, [previewData]);

  // Determine the best initial view mode once preview data is loaded
  useEffect(() => {
    if (previewData) {
      // Priority: PDF > Markdown > Unsupported
      if (canShowPdfPreview) {
        setViewMode('pdf');
      } else if (previewData.markdown_content) {
        setViewMode('markdown');
      } else {
        setViewMode('unsupported');
      }
    }
  }, [previewData, canShowPdfPreview]);

  // Fetch PDF file when view mode is switched to PDF
  useEffect(() => {
    if (viewMode === 'pdf' && canShowPdfPreview && !pdfFile && previewData) {
      const pdfPath = previewData.converted_pdf_object_path || previewData.doc_object_path;
      if (pdfPath) {
        fetchPdf(pdfPath);
      }
    }
  }, [viewMode, canShowPdfPreview, pdfFile, fetchPdf, previewData]);

  // Scroll to highlighted markdown line
  useEffect(() => {
    if (isHovering.current && viewMode === 'markdown' && highlightedChunk && highlightedChunk.metadata?.md_source_map) {
      const [start_line, end_line] = highlightedChunk.metadata.md_source_map;
      // lineRefs keys are 1-based, md_source_map is 0-based, so we add 1
      const startElement = lineRefs.current.get(start_line + 1);
      const endElement = lineRefs.current.get(end_line + 1);
      const containerElement = markdownContainerRef.current;

      if (startElement && endElement && containerElement) {
        const containerScrollTop = containerElement.scrollTop;
        const containerHeight = containerElement.clientHeight;

        const overallTop = startElement.offsetTop;
        const overallBottom = endElement.offsetTop + endElement.offsetHeight;
        const overallHeight = overallBottom - overallTop;

        const isTopVisible = overallTop >= containerScrollTop;
        const isBottomVisible = overallBottom <= containerScrollTop + containerHeight;

        if (isTopVisible && isBottomVisible) {
          return; // Already in view
        }

        let newScrollTop;
        if (overallHeight > containerHeight) {
          newScrollTop = overallTop;
        } else if (overallTop < containerScrollTop) {
          newScrollTop = overallTop;
        } else {
          newScrollTop = overallBottom - containerHeight;
        }

        containerElement.scrollTo({
          top: newScrollTop,
          behavior: 'smooth',
        });
      }
    }
  }, [highlightedChunk, viewMode]);

  const [pageDimensions, setPageDimensions] = useState(new Map());

  // Scroll to highlighted PDF page
  useEffect(() => {
    if (isHovering.current && viewMode === 'pdf' && highlightedChunk && highlightedChunk.metadata?.pdf_source_map) {
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

          const bboxTop = pageTopInContainer + (y1 * scale);
          const bboxBottom = pageTopInContainer + (y2 * scale);

          if (bboxTop < overallTop) overallTop = bboxTop;
          if (bboxBottom > overallBottom) overallBottom = bboxBottom;
        }
      }

      if (overallTop === Infinity) return; // No valid bboxes found

      const containerScrollTop = containerElement.scrollTop;
      const containerHeight = containerElement.clientHeight;
      const overallHeight = overallBottom - overallTop;

      const isTopVisible = overallTop >= containerScrollTop;
      const isBottomVisible = overallBottom <= containerScrollTop + containerHeight;

      if (isTopVisible && isBottomVisible) {
        return; // Already in view
      }

      let newScrollTop;
      if (overallHeight > containerHeight) {
        newScrollTop = overallTop;
      } else if (overallTop < containerScrollTop) {
        newScrollTop = overallTop;
      } else {
        newScrollTop = overallBottom - containerHeight;
      }

      containerElement.scrollTo({
        top: newScrollTop,
        behavior: 'smooth',
      });
    }
  }, [highlightedChunk, viewMode, pageDimensions, pdfWidth]);

  // Scroll chunk list when scrolling left panel
  useEffect(() => {
    if (scrolledToChunkId) {
      const chunkItem = chunkItemRefs.current.get(scrolledToChunkId);
      if (chunkItem) {
        chunkItem.scrollIntoView({
          behavior: 'smooth',
          block: 'start',
        });
      }
    }
  }, [scrolledToChunkId]);

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
    setPageDimensions((prev) => new Map(prev).set(page.pageNumber, { width: page.width, height: page.height }));
  };

  const renderPdfHighlight = (pageNumber: number) => {
    if (!highlightedChunk || !highlightedChunk.metadata?.pdf_source_map || viewMode !== 'pdf') {
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
        highlights.push(<div key={`${sourceMap.page_idx}-${x1}-${y1}`} style={style} />);
      }
    }
    return highlights;
  };

  const { run: handleScroll } = useDebounceFn(() => {
    if (isHovering.current) return;

    const container = viewMode === 'pdf' ? pdfContainerRef.current : markdownContainerRef.current;
    if (!container) return;

    const containerScrollTop = container.scrollTop;
    let focusChunkId = null;

    const chunks = previewData?.chunks || [];
    for (const chunk of chunks) {
      let top = Infinity;
      let bottom = -Infinity;

      if (viewMode === 'pdf' && chunk.metadata?.pdf_source_map) {
        chunk.metadata.pdf_source_map.forEach((sourceMap: any) => {
          const pageNumber = sourceMap.page_idx + 1;
          const pageElement = pageRefs.current.get(pageNumber);
          const pageDim = pageDimensions.get(pageNumber);
          if (pageElement && pageDim) {
            const [, y1, , y2] = sourceMap.bbox;
            const scale = (pdfWidth || pageDim.width) / pageDim.width;
            const pageTopInContainer = pageElement.offsetTop;
            const bboxTop = pageTopInContainer + (y1 * scale);
            const bboxBottom = pageTopInContainer + (y2 * scale);
            if (bboxTop < top) top = bboxTop;
            if (bboxBottom > bottom) bottom = bboxBottom;
          }
        });
      } else if (viewMode === 'markdown' && chunk.metadata?.md_source_map) {
        const [start_line, end_line] = chunk.metadata.md_source_map;
        const startElement = lineRefs.current.get(start_line + 1);
        const endElement = lineRefs.current.get(end_line + 1);
        if (startElement && endElement) {
          top = startElement.offsetTop;
          bottom = endElement.offsetTop + endElement.offsetHeight;
        }
      }

      if (top === Infinity) continue;

      // Find the first chunk whose bottom is below the top of the viewport
      if (bottom > containerScrollTop) {
        focusChunkId = chunk.id;
        break;
      }
    }

    // If scrolled to the very bottom, make sure the last chunk is selected
    if (container.scrollTop + container.clientHeight >= container.scrollHeight - 2) {
      if (chunks.length > 0) {
        focusChunkId = chunks[chunks.length - 1].id;
      }
    }

    if (focusChunkId && focusChunkId !== scrolledToChunkId) {
      setScrolledToChunkId(focusChunkId);
    }
  }, { wait: 150 });

  const renderMarkdownView = () => {
    if (!previewData?.markdown_content) {
      return <Empty description={<FormattedMessage id="chunk.viewer.markdown.empty" defaultMessage="No markdown content available for preview." />} />;
    }

    const lines = previewData.markdown_content.split('\n');
    const [start_line, end_line] = highlightedChunk?.metadata?.md_source_map || [null, null];

    return (
      <div ref={markdownContainerRef} onScroll={handleScroll} style={{ height: '80vh', overflowY: 'auto', border: '1px solid #f0f0f0', padding: '16px', fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>
        {lines.map((line: string, index: number) => {
          const lineNumber = index + 1;
          const isHighlighted = start_line !== null && end_line !== null && lineNumber >= start_line + 1 && lineNumber <= end_line + 1;
          return (
            <div
              key={lineNumber}
              ref={(el) => {
                if (el) lineRefs.current.set(lineNumber, el);
                else lineRefs.current.delete(lineNumber);
              }}
              className={isHighlighted ? styles.highlightedLine : ''}
            >
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {line || '\u00A0'}
              </ReactMarkdown>
            </div>
          );
        })}
      </div>
    );
  };

  const renderPdfView = () => {
    if (pdfLoading || !pdfFile) {
      return <div style={{ height: '80vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><Spin /></div>;
    }
    return (
      <div ref={pdfContainerRef} onScroll={handleScroll} style={{ height: '80vh', overflowY: 'auto', backgroundColor: '#f0f0f0' }}>
        <Document file={pdfFile} onLoadSuccess={onDocumentLoadSuccess}>
          {Array.from(new Array(numPages), (el, index) => (
            <div
              key={`page_${index + 1}`}
              ref={(el) => {
                if (el) pageRefs.current.set(index + 1, el);
                else pageRefs.current.delete(index + 1);
              }}
              style={{ display: 'flex', justifyContent: 'center', marginBottom: '8px' }}
            >
              <div style={{ position: 'relative' }}>
                <Page pageNumber={index + 1} width={pdfWidth} onRenderSuccess={onPageRenderSuccess} />
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
    return <Empty description={<FormattedMessage id="chunk.viewer.data.loadFailed" defaultMessage="Failed to load preview data." />} />;
  }

  const renderContent = () => {
    switch (viewMode) {
      case 'markdown':
        return <div key="markdown-view">{renderMarkdownView()}</div>;
      case 'pdf':
        return <div key="pdf-view">{renderPdfView()}</div>;
      default:
        return <div style={{ height: '80vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><Empty description={<FormattedMessage id="chunk.viewer.unsupported" defaultMessage="This document format is not supported for preview." />} /></div>;
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
        <Col span={12}>
          {renderContent()}
        </Col>
        <Col span={12}>
          <div ref={chunkListRef} style={{ height: '80vh', overflowY: 'auto' }}>
            <List
              header={<Typography.Title level={5}><FormattedMessage id="chunk.viewer.chunks.title" defaultMessage="Chunks" /></Typography.Title>}
              bordered
              dataSource={previewData?.chunks || []}
              renderItem={(item: Chunk) => {
                const isHighlightedByHover = highlightedChunk?.id === item.id;
                const backgroundColor = isHighlightedByHover ? '#e6f7ff' : 'transparent';

                return (
                  <List.Item
                    ref={(el) => {
                      if (el) chunkItemRefs.current.set(item.id, el);
                      else chunkItemRefs.current.delete(item.id);
                    }}
                    onMouseEnter={() => { isHovering.current = true; setHighlightedChunk(item); }}
                    onMouseLeave={() => { isHovering.current = false; setHighlightedChunk(null); }}
                    className={styles.chunkListItem}
                    style={{
                      cursor: 'pointer',
                      backgroundColor,
                      transition: 'background-color 0.3s ease',
                    }}
                  >
                    <div>
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {item.text}
                      </ReactMarkdown>
                    </div>
                  </List.Item>
                )
              }}
            />
          </div>
        </Col>
      </Row>
    </div>
  );
};
