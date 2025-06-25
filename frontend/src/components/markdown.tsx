import { CopyOutlined } from '@ant-design/icons';
import { GlobalToken, theme } from 'antd';
import 'highlight.js/styles/github-dark.css';
import Markdown from 'react-markdown';
import { toast } from 'react-toastify';
import rehypeHighlight from 'rehype-highlight';
import rehypeHighlightLines from 'rehype-highlight-code-lines';
import remarkDirective from 'remark-directive';
import remarkGfm from 'remark-gfm';
import remarkGithubAdmonitionsToDirectives from 'remark-github-admonitions-to-directives';
import { css, styled } from 'umi';
import { v4 as uuidV4 } from 'uuid';

type MarkdownProps = {
  children?: string;
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

export const ApeMarkdown = ({ children }: MarkdownProps) => {
  const { token } = theme.useToken();

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
        rehypePlugins={[rehypeHighlight, rehypeHighlightLines]}
        remarkPlugins={[
          remarkGfm,
          remarkGithubAdmonitionsToDirectives,
          remarkDirective,
        ]}
        components={{
          a: (props) => <a {...props} target="_blank" />,
          code: ({ className, children }) => {
            const match = /language-(\w+)/.exec(className || '');
            const id = uuidV4();
            if (match?.length) {
              return (
                <code id={id} className={className + ' copy-to-clipboard'}>
                  {/* @ts-ignore */}
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
        }}
      >
        {children}
      </Markdown>
    </StyledMarkdown>
  );
};
