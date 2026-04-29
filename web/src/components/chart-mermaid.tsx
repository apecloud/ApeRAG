'use client';
import { cn } from '@/lib/utils';
import { useTranslations } from 'next-intl';
import { useTheme } from 'next-themes';
import type { PanZoom } from 'panzoom';
import { useEffect, useId, useMemo, useRef, useState } from 'react';
import './chart-mermaid.css';
import { Card } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';

export const ChartMermaid = ({ children }: { children: string }) => {
  const [svg, setSvg] = useState('');
  const { resolvedTheme } = useTheme();
  const [error, setError] = useState<boolean>(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const reactId = useId();
  const id = useMemo(
    () => `mermaid-container-${reactId.replace(/[^a-zA-Z0-9_-]/g, '')}`,
    [reactId],
  );

  const components_dmermaid = useTranslations('components.dmermaid');

  const [tab, setTab] = useState<string>('graph');

  useEffect(() => {
    let cancelled = false;

    const renderMermaid = async () => {
      const isDark = resolvedTheme === 'dark';

      try {
        const { default: mermaid } = await import('mermaid');
        mermaid.initialize({
          startOnLoad: true,
          theme: isDark ? 'dark' : 'neutral',
          securityLevel: 'loose',
          themeVariables: {
            fontSize: 'inherit',
            labelBkg: 'transparent',
            lineColor: 'var(--input)',

            // Flowchart Variables
            nodeBorder: 'var(--border)',
            clusterBkg: 'var(--card)',
            clusterBorder: 'var(--input)',
            defaultLinkColor: 'var(--input)',
            edgeLabelBackground: 'transparent',
            titleColor: 'var(--muted-foreground)',
            nodeTextColor: 'var(--card-foreground)',
          },
          themeCSS: '.labelBkg { background: none; }',
          flowchart: {},
        });
        const { svg } = await mermaid.render(id, children);
        if (!cancelled) {
          setSvg(svg);
          setError(false);
        }
      } catch (err) {
        console.error(err);
        if (!cancelled) {
          setError(true);
        }
      }
    };

    renderMermaid();

    return () => {
      cancelled = true;
    };
  }, [children, id, resolvedTheme]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let disposed = false;
    let controller: PanZoom | undefined;

    import('panzoom').then(({ default: createPanZoom }) => {
      if (disposed) return;
      controller = createPanZoom(container, {
        minZoom: 0.5,
        maxZoom: 5,
      });
    });

    return () => {
      disposed = true;
      controller?.dispose();
    };
  }, []);

  return (
    <>
      <Tabs value={tab} className="font-sans" onValueChange={setTab}>
        <TabsList className="w-full">
          <TabsTrigger value="graph">
            {components_dmermaid('graph')}
          </TabsTrigger>
          <TabsTrigger value="data">{components_dmermaid('data')}</TabsTrigger>
        </TabsList>
        <TabsContent
          value="graph"
          forceMount
          className={tab === 'graph' ? 'block' : 'hidden'}
        >
          <Card className="my-2 min-h-80 cursor-move overflow-hidden rounded-md p-4">
            <div
              ref={containerRef}
              data-error={error}
              className={`${id} flex justify-center`}
              dangerouslySetInnerHTML={{
                __html: svg,
              }}
            />
          </Card>
        </TabsContent>
        <TabsContent value="data">
          <code className={cn('hljs language-mermaid my-2 rounded-md text-sm')}>
            {children}
          </code>
        </TabsContent>
      </Tabs>
    </>
  );
};
