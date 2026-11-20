import { lazy, memo, Suspense, useState, type ReactNode } from "react";
import { Check, ChevronLeft, ChevronRight, Copy, FastForward, Pencil, RefreshCw } from "lucide-react";
import { useTranslation } from "react-i18next";
import { GenerationTrace } from "@/components/chat/generation-trace";
import { Button } from "@/components/ui/button";
import { useFormat } from "@/hooks/use-format";
import type { ChatMessage } from "@/lib/api";
import type { ActiveGeneration } from "@/lib/generation";

const MarkdownContent = lazy(() => import("@/components/markdown-content").then((module) => ({ default: module.MarkdownContent })));

type MessageItemProps = {
  message: ChatMessage;
  streaming: ActiveGeneration | null;
  isLast: boolean;
  busy: boolean;
  onEdit: (message: ChatMessage, content: string) => void;
  onRegenerate: (message: ChatMessage) => void;
  onContinue: (message: ChatMessage) => void;
  onUseVariant: (message: ChatMessage, index: number) => void;
};

function IconAction({ label, onClick, disabled, children }: { label: string; onClick: () => void; disabled?: boolean; children: ReactNode }) {
  return (
    <Button variant="ghost" size="icon" className="h-7 w-7" aria-label={label} title={label} disabled={disabled} onClick={onClick}>{children}</Button>
  );
}

function UserMessage({ message, busy, onEdit }: Pick<MessageItemProps, "message" | "busy" | "onEdit">) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(message.content);
  return (
    <article className="group flex flex-col items-end gap-0.5" data-role="user">
      {editing ? (
        <div className="w-full rounded-2xl border bg-background p-2 shadow-sm">
          <textarea aria-label={t("chat.edit")} autoFocus className="block min-h-20 w-full resize-y bg-transparent p-2 text-[14px] focus-visible:ring-0 focus-visible:ring-offset-0" value={draft} onChange={(event) => setDraft(event.target.value)} />
          <div className="flex justify-end gap-2">
            <Button size="sm" variant="ghost" onClick={() => { setEditing(false); setDraft(message.content); }}>{t("common.cancel")}</Button>
            <Button size="sm" disabled={!draft.trim() || busy} onClick={() => { setEditing(false); onEdit(message, draft.trim()); }}>{t("chat.saveAndSend")}</Button>
          </div>
        </div>
      ) : (
        <>
          <div className="max-w-[85%] whitespace-pre-wrap break-words rounded-2xl bg-muted px-3.5 py-2 text-[14px] leading-6">{message.content}</div>
          <div className="flex items-center opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
            <IconAction label={copied ? t("common.copied") : t("chat.copy")} onClick={() => { void navigator.clipboard.writeText(message.content); setCopied(true); setTimeout(() => setCopied(false), 1500); }}>{copied ? <Check size={13} /> : <Copy size={13} />}</IconAction>
            <IconAction label={t("chat.edit")} disabled={busy} onClick={() => { setDraft(message.content); setEditing(true); }}><Pencil size={13} /></IconAction>
          </div>
        </>
      )}
    </article>
  );
}

function AssistantMessage({ message, streaming, isLast, busy, onRegenerate, onContinue, onUseVariant }: Omit<MessageItemProps, "onEdit">) {
  const { t } = useTranslation();
  const format = useFormat();
  const [copied, setCopied] = useState(false);
  const total = message.variants.length + 1;
  const [view, setView] = useState<number | null>(null);
  const viewing = view !== null && view < message.variants.length ? message.variants[view] : null;
  const shown = viewing ?? message;
  const content = streaming ? streaming.text : shown.content;
  const stages = streaming ? streaming.stages : viewing ? [] : message.reasoning_trace;
  const inProgress = streaming !== null && ["queued", "running"].includes(streaming.status);
  const tps = streaming?.tokensPerSecond ?? shown.tokens_per_second;
  const outputTokens = streaming?.outputTokens ?? shown.output_tokens;
  const first = streaming?.firstTokenMs ?? null;
  const model = shown.checkpoint_id ?? shown.model_id;
  const position = (view ?? message.variants.length) + 1;
  const move = (delta: number) => setView(Math.min(Math.max((view ?? message.variants.length) + delta, 0), total - 1));
  return (
    <article className="group" data-role="assistant">
      <GenerationTrace events={stages} active={streaming} />
      {content ? (
        <div className={inProgress ? "streaming-text" : undefined}>
          <Suspense fallback={<p className="whitespace-pre-wrap text-[15px] leading-7">{content}</p>}><MarkdownContent content={content} /></Suspense>
          {inProgress ? <span aria-hidden className="ml-0.5 inline-block h-4 w-[3px] translate-y-0.5 rounded-full bg-accent animate-caret" /> : null}
        </div>
      ) : inProgress ? (
        <span aria-hidden className="inline-block h-4 w-[3px] rounded-full bg-accent animate-caret" />
      ) : (
        <p className="text-[13px] text-muted-foreground">{message.status === "failed" ? t("chat.failedEmpty") : t("chat.emptyAnswer")}</p>
      )}
      {message.status === "cancelled" || message.status === "interrupted" ? <p className="mt-1 text-xs text-warning">{t("chat.partial")}</p> : null}
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
        {!inProgress ? (
          <div className="-ml-1.5 flex items-center opacity-80 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
            {total > 1 ? (
              <>
                <IconAction label={t("chat.prevVariant")} disabled={position <= 1} onClick={() => move(-1)}><ChevronLeft size={14} /></IconAction>
                <span className="px-0.5 tabular-nums" aria-live="polite">{position} / {total}</span>
                <IconAction label={t("chat.nextVariant")} disabled={position >= total} onClick={() => move(1)}><ChevronRight size={14} /></IconAction>
              </>
            ) : null}
            <IconAction label={copied ? t("common.copied") : t("chat.copy")} onClick={() => { void navigator.clipboard.writeText(content); setCopied(true); setTimeout(() => setCopied(false), 1500); }}>{copied ? <Check size={13} /> : <Copy size={13} />}</IconAction>
            {isLast && !viewing ? (
              <>
                <IconAction label={t("chat.regenerate")} disabled={busy} onClick={() => { setView(null); onRegenerate(message); }}><RefreshCw size={13} /></IconAction>
                <IconAction label={t("chat.continue")} disabled={busy} onClick={() => onContinue(message)}><FastForward size={13} /></IconAction>
              </>
            ) : null}
            {viewing ? <Button size="sm" variant="outline" className="ml-1" disabled={busy} onClick={() => { onUseVariant(message, view ?? 0); setView(null); }}>{t("chat.useVersion")}</Button> : null}
          </div>
        ) : null}
        {model ? <span className="truncate" title={model}>{model.split("/").slice(-2).join("/")}</span> : null}
        {tps ? (
          <span className="tabular-nums">{format.number(tps, 1)} {t("units.tokensPerSecond")}</span>
        ) : !inProgress && shown.generation_ms !== null ? (
          <span className="tabular-nums">{t("chat.responseTime", { value: format.number(shown.generation_ms / 1000, 2) })}</span>
        ) : null}
        {!inProgress && outputTokens ? <span className="tabular-nums">{t("chat.tokens", { count: outputTokens })}</span> : null}
        {first ? <span className="tabular-nums">{t("chat.ttft", { value: format.number(first / 1000, 2) })}</span> : null}
      </div>
    </article>
  );
}

function MessageItemBase(props: MessageItemProps) {
  return props.message.role === "user" ? <UserMessage message={props.message} busy={props.busy} onEdit={props.onEdit} /> : <AssistantMessage {...props} />;
}

export const MessageItem = memo(MessageItemBase);
