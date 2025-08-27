'use client';
import mermaid from 'mermaid';
import { useTheme } from 'next-themes';
import { useCallback, useEffect, useState } from 'react';
import './chart-mermaid.css';

export const ChartMermaid = ({ children }: { children: string }) => {
  const [svg, setSvg] = useState('');
  const { resolvedTheme } = useTheme();
  const [error, setError] = useState<boolean>(false);

  const [id, setId] = useState<string>();

  const renderMermaid = useCallback(async () => {
    const isDark = resolvedTheme === 'dark';

    try {
      mermaid.initialize({
        startOnLoad: true,
        theme: isDark ? 'dark' : 'neutral',
        securityLevel: 'loose',
        themeVariables: {
          // primaryColor: '#0165ca',
          // primaryTextColor: '#fff',
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
      const { svg } = await mermaid.render(`mermaid-container-${id}`, children);
      setSvg(svg);
      setError(false);
    } catch (err) {
      console.log(err);
      setError(true);
    }
  }, [children, id, resolvedTheme]);

  useEffect(() => {
    renderMermaid();
  }, [renderMermaid]);

  useEffect(() => {
    setId(String((Math.random() * 100000).toFixed(0)));
  }, []);

  return (
    <div
      data-error={error}
      className={`mermaid-container-${id} my-4 flex justify-center`}
      dangerouslySetInnerHTML={{
        __html: svg,
      }}
    />
  );
};
