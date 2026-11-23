import { isValidElement, memo, useMemo, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import "katex/dist/katex.min.css";
import { Check, Copy } from "lucide-react";
import { splitBlocks } from "@/lib/markdown-blocks";

const REMARK = [remarkGfm, remarkMath];
const REHYPE: NonNullable<React.ComponentProps<typeof ReactMarkdown>["rehypePlugins"]> = [rehypeKatex, [rehypeHighlight, { detect: true, ignoreMissing: true }]];

function nodeText(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(nodeText).join("");
  if (isValidElement<{ children?: ReactNode }>(node)) return nodeText(node.props.children);
  return "";
}

function CodeBlock({ children, className }: { children?: ReactNode; className?: string }) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);
  const text = nodeText(children).replace(/\n$/, "");
  const language = /language-(\w+)/.exec(className ?? "")?.[1] ?? "";
  if (!className && !text.includes("\n")) {
    return <code className="rounded bg-muted px-1 py-0.5 font-mono text-[0.88em]">{children}</code>;
  }
  return (
    <div className="my-4 overflow-hidden rounded-lg border hairline bg-muted/40">
      <div className="flex items-center justify-between px-3 py-1.5 text-xs text-muted-foreground">
        <span className="font-mono">{language || "code"}</span>
        <button
          type="button"
          className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 hover:bg-muted hover:text-foreground"
          onClick={() => {
            void navigator.clipboard.writeText(text);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          }}
        >
          {copied ? <Check size={12} /> : <Copy size={12} />}
          <span>{copied ? t("common.copied") : t("chat.copyCode")}</span>
        </button>
      </div>
      <pre className="overflow-x-auto px-3 pb-3 font-mono text-[13px] leading-relaxed">
        <code className={className}>{children}</code>
      </pre>
    </div>
  );
}

const COMPONENTS: Components = { code: ({ className, children }) => <CodeBlock className={className}>{children}</CodeBlock> };

const Block = memo(function Block({ text }: { text: string }) {
  return (
    <ReactMarkdown remarkPlugins={REMARK} rehypePlugins={REHYPE} components={COMPONENTS}>
      {text}
    </ReactMarkdown>
  );
});

const PROSE =
  "max-w-none break-words text-[15px] leading-7 text-foreground [&_a]:text-accent [&_a]:underline [&_blockquote]:my-3 [&_blockquote]:border-l-2 [&_blockquote]:border-border [&_blockquote]:pl-4 [&_blockquote]:text-muted-foreground [&_h1]:mb-2 [&_h1]:mt-6 [&_h1]:text-xl [&_h1]:font-semibold [&_h2]:mb-2 [&_h2]:mt-5 [&_h2]:text-lg [&_h2]:font-semibold [&_h3]:mb-1.5 [&_h3]:mt-4 [&_h3]:text-base [&_h3]:font-semibold [&_hr]:my-6 [&_li]:my-1 [&_ol]:my-3 [&_ol]:list-decimal [&_ol]:pl-6 [&_p]:my-3 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0 [&_table]:my-4 [&_table]:w-full [&_table]:border-collapse [&_td]:border-b [&_td]:px-3 [&_td]:py-2 [&_th]:border-b [&_th]:px-3 [&_th]:py-2 [&_th]:text-left [&_th]:font-medium [&_ul]:my-3 [&_ul]:list-disc [&_ul]:pl-6";

export function MarkdownContent({ content }: { content: string }) {
  const blocks = useMemo(() => splitBlocks(content), [content]);
  return (
    <div className={PROSE}>
      {blocks.map((block, index) => (
        <Block key={index} text={block} />
      ))}
    </div>
  );
}
