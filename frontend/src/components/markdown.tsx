import { CaretRightOutlined, CopyOutlined } from '@ant-design/icons';
import { Collapse, GlobalToken, Space, theme } from 'antd';
import 'highlight.js/styles/github-dark.css';
import { useMemo } from 'react';
import Markdown from 'react-markdown';
import { toast } from 'react-toastify';
import rehypeHighlight from 'rehype-highlight';
import rehypeHighlightLines from 'rehype-highlight-code-lines';
import rehypeRaw from 'rehype-raw';
import remarkDirective from 'remark-directive';
import remarkGfm from 'remark-gfm';
import remarkGithubAdmonitionsToDirectives from 'remark-github-admonitions-to-directives';
import { css, styled } from 'umi';
import { v4 as uuidV4 } from 'uuid';
import { AuthAssetImage } from './AuthAssetImage';

type MarkdownProps = {
  children?: string;
  isAgent?: boolean;
};

const StyledMarkdown = styled('div').withConfig({
  shouldForwardProp: (prop) => !['token'].includes(prop),
})<{
  token: GlobalToken;
}>`
  ${({ token }) => {
    return css`
      line-height: 1.8em;
      pre {
        background: #0d1117;
        color: #c9d1d9;
        border-radius: 4px;
        font-size: 12px;
      }
      code {
        background: rgba(128, 128, 128, 0.15);
        padding-inline: 8px;
        padding-block: 4px;
        border-radius: 4px;
        word-wrap: break-word;
      }
      pre > code {
        display: block;
        overflow-x: auto;
        padding: 1em;
        border-radius: 4px;
        font-size: 12px;
      }

      img {
        max-width: 100%;
      }
      hr {
        border-color: ${token.colorBorderSecondary};
        border-style: solid;
        border-width: 1px;
        border-top: none;
      }
      table {
        display: block;
        overflow: auto;
        margin-block: 24px;
      }
      table th,
      table td {
        border: 1px solid ${token.colorBorderSecondary};
        text-align: left;
        padding: 12px;
      }
      table th p,
      table td p {
        margin: 0;
      }
      .copy-to-clipboard {
        position: relative;
        .copy-to-clipboard-trigger {
          display: none;
        }
        &:hover .copy-to-clipboard-trigger {
          display: block;
        }
      }

      .ape-collapse {
        margin-bottom: 12px;
      }
      .ape-collapse-content-box {
        border: 1px dashed ${token.colorBorderSecondary};
        background: ${token.colorBgLayout};
        padding: 12px !important;
        border-radius: 4px;
      }
    `;
  }}
`;

const StyledCopyButton = styled('div').withConfig({
  shouldForwardProp: (prop) => !['token'].includes(prop),
})<{
  token: GlobalToken;
}>`
  ${({ token }) => {
    return css`
      position: absolute;
      right: 0.5em;
      top: 0.5em;
      width: 32px;
      height: 32px;
      border-radius: 4px;
      border: 1px solid rgba(255, 255, 255, 0.3);
      color: rgba(255, 255, 255, 0.6);
      cursor: pointer;
      text-align: center;
      line-height: 32px;
      background: #0d1116;
      transition:
        border-color 0.3s,
        color 0.3s;
      &:hover {
        color: #fff;
        border-color: ${token.colorPrimary};
      }
    `;
  }}
`;

export const CollapseResult = ({
  title,
  className,
  children,
}: {
  title: string;
  className?: string;
  children?: string;
}) => {
  return (
    <Collapse
      expandIcon={({ isActive }) => (
        <CaretRightOutlined rotate={isActive ? 90 : 0} />
      )}
      defaultActiveKey={['conent']}
      style={{ background: 'none' }}
      bordered={false}
      className={className}
      items={[
        {
          key: 'conent',
          label: <Space>💡 {title}</Space>,
          children,
        },
      ]}
    />
  );
};

export const ApeMarkdown = ({ children, isAgent = false }: MarkdownProps) => {
  const { token } = theme.useToken();

  let processedValue = useMemo(() => {
    return children
      ?.replace(/<think>/g, '<div class="think">')
      .replace(/<\/think>/g, '</div>')
      .replace(/<tool_call_result>/g, '<div class="tool_call_result">')
      .replace(/<\/tool_call_result>/g, '</div>');
  }, [children]);

  const onCopy = async (id: string) => {
    const text = document.getElementById(id)?.innerText || '';
    if (text) {
      await navigator.clipboard.writeText(text);
      toast.success('Copied!');
    }
  };

  return (
    <StyledMarkdown token={token}>
      <Markdown
        rehypePlugins={[rehypeHighlight, rehypeHighlightLines, rehypeRaw]}
        remarkPlugins={[
          remarkGfm,
          remarkGithubAdmonitionsToDirectives,
          remarkDirective,
        ]}
        urlTransform={(url) =>
          url.startsWith('asset://')
            ? url
            : new URL(url, window.location.href).href
        }
        components={{
          img: (props) => {
            if (props.src?.startsWith('asset://')) {
              return (
                <AuthAssetImage
                  src={props.src}
                  style={isAgent ? { maxHeight: 400 } : {}}
                />
              );
            }
            return <img {...props} />;
          },
          a: (props) => <a {...props} target="_blank" />,
          code: ({ className, children }) => {
            const match = /language-(\w+)/.exec(className || '');
            const id = uuidV4();
            if (match?.length) {
              return (
                <code id={id} className={className + ' copy-to-clipboard'}>
                  <StyledCopyButton
                    className="copy-to-clipboard-trigger"
                    token={token}
                    onClick={() => onCopy(id)}
                  >
                    <CopyOutlined />
                  </StyledCopyButton>
                  {children}
                </code>
              );
            } else {
              return <code>{children}</code>;
            }
          },
          div: (props: any) => {
            const className = props?.className || '';
            const children = props?.children || '';
            if (/think/.exec(className)?.length) {
              return (
                <CollapseResult title="Thinking" className={className}>
                  {children}
                </CollapseResult>
              );
            }
            if (/tool_call_result/.exec(className)?.length) {
              return (
                <CollapseResult title="Tool call" className={className}>
                  {children}
                </CollapseResult>
              );
            }
            return <div {...props} />;
          },
        }}
      >
        {processedValue}
      </Markdown>
    </StyledMarkdown>
  );
};
