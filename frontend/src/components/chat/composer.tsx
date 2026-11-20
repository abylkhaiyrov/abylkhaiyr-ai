import { useEffect, useRef, useState } from "react";
import { ArrowUp, SlidersHorizontal, Square } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { ModelPicker } from "@/components/chat/model-picker";
import type { Model, ModelRegistry } from "@/lib/api";
import { useFormat } from "@/hooks/use-format";

type ComposerProps = {
  registry: ModelRegistry | undefined;
  model: Model | null;
  generating: boolean;
  offline: boolean;
  contextUsed: number | null;
  contextLimit: number | null;
  onSend: (content: string) => void;
  onStop: () => void;
  onModel: (model: Model) => void;
  onSettings: () => void;
  focusKey: string | null;
};

export function Composer({ registry, model, generating, offline, contextUsed, contextLimit, onSend, onStop, onModel, onSettings, focusKey }: ComposerProps) {
  const { t } = useTranslation();
  const format = useFormat();
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 200)}px`;
  }, [value]);
  useEffect(() => ref.current?.focus(), [focusKey]);
  const canSend = value.trim().length > 0 && !generating && !offline && model !== null;
  const submit = () => {
    if (!canSend) return;
    onSend(value.trim());
    setValue("");
  };
  return (
    <div className="rounded-2xl border bg-card p-2 shadow-sm focus-within:border-primary/60">
      <textarea
        ref={ref}
        rows={1}
        aria-label={t("chat.prompt")}
        placeholder={t("chat.prompt")}
        className="block max-h-[200px] w-full resize-none bg-transparent px-2 py-1.5 text-sm placeholder:text-muted-foreground focus-visible:ring-0 focus-visible:ring-offset-0"
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
            event.preventDefault();
            submit();
          }
        }}
      />
      <div className="mt-1 flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-1.5">
          <ModelPicker registry={registry} selected={model} locked={generating} onSelect={onModel} />
          <Button variant="ghost" size="icon" className="h-8 w-8" aria-label={t("chat.settings")} title={t("chat.settings")} onClick={onSettings}><SlidersHorizontal size={15} /></Button>
          {contextUsed !== null && contextLimit ? <span className="hidden text-xs text-muted-foreground tabular-nums sm:inline">{t("chat.context", { used: format.number(contextUsed, 0), limit: format.number(contextLimit, 0) })}</span> : null}
        </div>
        {generating ? (
          <Button variant="danger" size="icon" className="h-8 w-8 rounded-full" aria-label={t("chat.stop")} onClick={onStop}><Square size={13} fill="currentColor" /></Button>
        ) : (
          <Button size="icon" className="h-8 w-8 rounded-full" aria-label={t("chat.send")} disabled={!canSend} onClick={submit}><ArrowUp size={16} /></Button>
        )}
      </div>
    </div>
  );
}
