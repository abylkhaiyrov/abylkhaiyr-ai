import { lazy, memo, Suspense, useState } from "react";
import { Check, Copy, FastForward, Pencil, RefreshCw } from "lucide-react";
import { useTranslation } from "react-i18next";
const MarkdownContent = lazy(() => import("@/components/markdown-content").then((module) => ({ default: module.MarkdownContent })));
import { GenerationTrace } from "@/components/chat/generation-trace";
import { Button } from "@/components/ui/button";
import { useFormat } from "@/hooks/use-format";
import type { ChatMessage } from "@/lib/api";
import type { ActiveGeneration } from "@/lib/generation";
import { cn } from "@/lib/utils";

type MessageItemProps = {
  message: ChatMessage;
  streaming: ActiveGeneration | null;
  isLast: boolean;
  busy: boolean;
  onEdit: (message: ChatMessage, content: string) => void;
  onRegenerate: (message: ChatMessage) => void;
  onContinue: (message: ChatMessage) => void;
};

function IconAction({ label, onClick, disabled, children }: { label: string; onClick: () => void; disabled?: boolean; children: React.ReactNode }) {
  return (
    <Button variant="ghost" size="icon" className="h-7 w-7 text-muted-foreground" aria-label={label} title={label} disabled={disabled} onClick={onClick}>{children}</Button>
  );
}

function MessageItemBase({ message, streaming, isLast, busy, onEdit, onRegenerate, onContinue }: MessageItemProps) {
  const { t } = useTranslation();
  const format = useFormat();
  const [copied, setCopied] = useState(false);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(message.content);
  const content = streaming ? streaming.text : message.content;
  const copy = () => {
    void navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  if (message.role === "user") {
    return (
      <article className="group flex flex-col items-end gap-1" data-role="user">
        {editing ? (
          <div className="w-full max-w-[85%] rounded-2xl border bg-card p-2">
            <textarea aria-label={t("chat.edit")} autoFocus className="field min-h-20 resize-y border-0 p-2" value={draft} onChange={(event) => setDraft(event.target.value)} />
            <div className="mt-1 flex justify-end gap-2">
              <Button size="sm" variant="ghost" onClick={() => { setEditing(false); setDraft(message.content); }}>{t("common.cancel")}</Button>
              <Button size="sm" disabled={!draft.trim() || busy} onClick={() => { setEditing(false); onEdit(message, draft.trim()); }}>{t("chat.saveAndSend")}</Button>
            </div>
          </div>
        ) : (
          <>
            <div className="max-w-[85%] whitespace-pre-wrap break-words rounded-2xl rounded-br-md bg-primary px-3.5 py-2 text-sm text-primary-foreground">{message.content}</div>
            <div className="flex items-center opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
              <IconAction label={copied ? t("common.copied") : t("chat.copy")} onClick={copy}>{copied ? <Check size={14} /> : <Copy size={14} />}</IconAction>
              <IconAction label={t("chat.edit")} disabled={busy} onClick={() => { setDraft(message.content); setEditing(true); }}><Pencil size={14} /></IconAction>
            </div>
          </>
        )}
      </article>
    );
  }

  const stages = streaming ? streaming.stages : message.reasoning_trace;
  const inProgress = streaming !== null && ["queued", "running"].includes(streaming.status);
  const first = streaming?.firstTokenMs ?? (message.generation_ms !== null ? null : null);
  const tps = streaming?.tokensPerSecond ?? message.tokens_per_second;
  const outputTokens = streaming?.outputTokens ?? message.output_tokens;
  const model = message.checkpoint_id ?? message.model_id;
  return (
    <article className="group grid gap-1" data-role="assistant">
      <GenerationTrace events={stages} active={streaming} />
      {content ? (
        <div className={cn(inProgress && "streaming-text")}>
          <Suspense fallback={<p className="whitespace-pre-wrap text-sm">{content}</p>}><MarkdownContent content={content} /></Suspense>
          {inProgress ? <span aria-hidden className="ml-0.5 inline-block h-4 w-1.5 translate-y-0.5 rounded-sm bg-primary animate-caret" /> : null}
        </div>
      ) : inProgress ? (
        <span aria-hidden className="inline-block h-4 w-1.5 rounded-sm bg-primary animate-caret" />
      ) : (
        <p className="text-sm text-muted-foreground">{message.status === "failed" ? t("chat.failedEmpty") : t("chat.emptyAnswer")}</p>
      )}
      {message.status === "cancelled" || message.status === "interrupted" ? <p className="text-xs text-warning">{t("chat.partial")}</p> : null}
      {!inProgress ? (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
          <div className="-ml-1.5 flex items-center opacity-70 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
            <IconAction label={copied ? t("common.copied") : t("chat.copy")} onClick={copy}>{copied ? <Check size={14} /> : <Copy size={14} />}</IconAction>
            {isLast ? (
              <>
                <IconAction label={t("chat.regenerate")} disabled={busy} onClick={() => onRegenerate(message)}><RefreshCw size={14} /></IconAction>
                <IconAction label={t("chat.continue")} disabled={busy} onClick={() => onContinue(message)}><FastForward size={14} /></IconAction>
              </>
            ) : null}
          </div>
          {model ? <span className="truncate" title={model}>{model.split("/").slice(-2).join("/")}</span> : null}
          {tps ? <span className="tabular-nums">{format.number(tps, 1)} {t("units.tokensPerSecond")}</span> : null}
          {outputTokens ? <span className="tabular-nums">{t("chat.tokens", { count: outputTokens })}</span> : null}
          {first ? <span className="tabular-nums">{t("chat.ttft", { value: format.number(first / 1000, 2) })}</span> : null}
        </div>
      ) : (
        <div className="flex gap-3 text-xs text-muted-foreground tabular-nums">
          {tps ? <span>{format.number(tps, 1)} {t("units.tokensPerSecond")}</span> : null}
          {first ? <span>{t("chat.ttft", { value: format.number(first / 1000, 2) })}</span> : null}
        </div>
      )}
    </article>
  );
}

export const MessageItem = memo(MessageItemBase);
