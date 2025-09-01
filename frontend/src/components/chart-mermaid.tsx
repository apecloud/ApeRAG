'use client';

import mermaid from 'mermaid';

import { Card, Tabs, theme } from 'antd';
import panzoom from 'panzoom';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useModel } from 'umi';
import './chart-mermaid.css';

export const ChartMermaid = ({ children }: { children: string }) => {
  const [svg, setSvg] = useState('');

  const [error, setError] = useState<boolean>(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const [id, setId] = useState<string>();
  const { themeName, setThemeName } = useModel('global');
  const { token } = theme.useToken();
  const [tab, setTab] = useState<string>('graph');

  const renderMermaid = useCallback(async () => {
    const isDark = themeName === 'dark';

    try {
      mermaid.initialize({
        startOnLoad: true,
        theme: isDark ? 'dark' : 'neutral',
        securityLevel: 'loose',
        // themeVariables: {
        //   // primaryColor: '#0165ca',
        //   // primaryTextColor: '#fff',
        //   fontSize: 'inherit',
        //   labelBkg: 'transparent',
        //   lineColor: 'var(--input)',

        //   // Flowchart Variables
        //   nodeBorder: 'var(--border)',
        //   clusterBkg: 'var(--card)',
        //   clusterBorder: 'var(--input)',
        //   defaultLinkColor: 'var(--input)',
        //   edgeLabelBackground: 'transparent',
        //   titleColor: 'var(--muted-foreground)',
        //   nodeTextColor: 'var(--card-foreground)',
        // },
        // themeCSS: '.labelBkg { background: none; }',
        flowchart: {},
      });
      const { svg } = await mermaid.render(`mermaid-container-${id}`, children);
      setSvg(svg);
      setError(false);
    } catch (err) {
      console.log(err);
      setError(true);
    }
  }, [children, id]);

  useEffect(() => {
    renderMermaid();
  }, [renderMermaid]);

  useEffect(() => {
    setId(String((Math.random() * 100000).toFixed(0)));
  }, []);

  useEffect(() => {
    if (containerRef.current) {
      panzoom(containerRef.current, {
        minZoom: 0.5,
        maxZoom: 5,
      });
    }
  }, []);

  return (
    <>
      <div style={{ background: token.colorBgContainer }}>
        <Tabs
          defaultActiveKey={tab}
          onChange={setTab}
          items={[
            {
              key: 'graph',
              label: 'Graph',
            },
            {
              key: 'data',
              label: 'Data',
            },
          ]}
        />
        <Card
          style={{
            display: tab === 'graph' ? 'block' : 'none',
            minHeight: 320,
            overflow: 'auto',
          }}
        >
          <div
            ref={containerRef}
            data-error={error}
            className={`mermaid-container-${id} flex justify-center`}
            dangerouslySetInnerHTML={{
              __html: svg,
            }}
          />
        </Card>
      </div>
      <div style={{ display: tab === 'data' ? 'block' : 'none' }}>
        <code>{children}</code>
      </div>
    </>
  );
};
