import { useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { Check, Copy } from "lucide-react";

function CodeBlock({ children, className }: { children?: ReactNode; className?: string }) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);
  const text = String(children ?? "").replace(/\n$/, "");
  const language = className?.replace("language-", "") ?? "";
  if (!className && !text.includes("\n")) {
    return (
      <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-[0.85em] text-foreground">
        {children}
      </code>
    );
  }
  return (
    <div className="my-3 overflow-hidden rounded-lg border bg-muted/50">
      <div className="flex items-center justify-between border-b bg-muted/80 px-3 py-1.5 text-[11px] text-muted-foreground">
        <span className="font-mono">{language || "code"}</span>
        <button
          type="button"
          className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 transition-colors hover:bg-background hover:text-foreground"
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
      <pre className="overflow-x-auto p-3 font-mono text-xs leading-relaxed">
        <code>{text}</code>
      </pre>
    </div>
  );
}

export function MarkdownContent({ content }: { content: string }) {
  return (
    <div className="prose-sm max-w-none break-words text-sm leading-relaxed text-foreground [&_blockquote]:my-2 [&_blockquote]:border-l-2 [&_blockquote]:border-primary/40 [&_blockquote]:pl-3 [&_blockquote]:text-muted-foreground [&_h1]:my-3 [&_h1]:text-lg [&_h1]:font-semibold [&_h2]:my-2.5 [&_h2]:text-base [&_h2]:font-semibold [&_h3]:my-2 [&_h3]:text-sm [&_h3]:font-semibold [&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5 [&_p]:my-2 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0 [&_table]:my-3 [&_table]:w-full [&_table]:border-collapse [&_td]:border [&_td]:px-2.5 [&_td]:py-1.5 [&_th]:border [&_th]:bg-muted/60 [&_th]:px-2.5 [&_th]:py-1.5 [&_th]:text-left [&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          code: ({ className, children }) => (
            <CodeBlock className={className}>{children}</CodeBlock>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
