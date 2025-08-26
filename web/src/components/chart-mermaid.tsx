'use client';
import mermaid from 'mermaid';
import { useTheme } from 'next-themes';
import { useCallback, useEffect, useState } from 'react';

export const ChartMermaid = ({ children }: { children: string }) => {
  const [svg, setSvg] = useState('');
  const [error, setError] = useState('');
  const { resolvedTheme } = useTheme();

  const renderMermaid = useCallback(async () => {
    try {
      mermaid.initialize({
        startOnLoad: true,
        theme: resolvedTheme === 'dark' ? 'dark' : 'neutral',
        securityLevel: 'loose',
      });
      const { svg } = await mermaid.render('mermaid-container', children);
      setSvg(svg);
      setError('');
    } catch (err) {
      setError('render error');
      console.log(err);
    }
  }, [children, resolvedTheme]);

  useEffect(() => {
    renderMermaid();
  }, [renderMermaid]);

  if (error) {
    return <div>{error}</div>;
  }

  return (
    <div
      className="mermaid-container"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
};
