'use client';
import mermaid from 'mermaid';
import { useTheme } from 'next-themes';
import { useCallback, useEffect, useMemo, useState } from 'react';
import './chart-mermaid.css';

export const ChartMermaid = ({ children }: { children: string }) => {
  const [svg, setSvg] = useState('');
  const { resolvedTheme } = useTheme();
  const [error, setError] = useState<boolean>(false);
  const id = useMemo(() => (Math.random() * 100000).toFixed(0), []);

  const renderMermaid = useCallback(async () => {
    try {
      mermaid.initialize({
        startOnLoad: true,
        theme: resolvedTheme === 'dark' ? 'dark' : 'neutral',
        securityLevel: 'loose',
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

  return (
    <div
      data-error={error}
      className={`mermaid-container-${id}`}
      dangerouslySetInnerHTML={{
        __html: svg,
      }}
    />
  );
};
