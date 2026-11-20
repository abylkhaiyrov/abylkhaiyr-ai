import { useEffect, useRef, useState } from "react";
import { ArrowUp, Square } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";

type ComposerProps = {
  generating: boolean;
  disabled: boolean;
  disabledReason: string | null;
  onSend: (content: string) => void;
  onStop: () => void;
  focusKey: string | null;
  large?: boolean;
};

export function Composer({ generating, disabled, disabledReason, onSend, onStop, focusKey, large = false }: ComposerProps) {
  const { t } = useTranslation();
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 220)}px`;
  }, [value]);
  useEffect(() => ref.current?.focus(), [focusKey]);
  const canSend = value.trim().length > 0 && !generating && !disabled;
  const submit = () => {
    if (!canSend) return;
    onSend(value.trim());
    setValue("");
  };
  return (
    <div>
      <div className="flex items-end gap-2 rounded-2xl border bg-background px-3 py-2 shadow-sm transition-colors focus-within:border-muted-foreground/50">
        <textarea
          ref={ref}
          rows={large ? 2 : 1}
          aria-label={t("chat.prompt")}
          placeholder={t("chat.prompt")}
          className="block max-h-[220px] min-h-[28px] w-full resize-none bg-transparent py-1 text-[15px] leading-6 placeholder:text-muted-foreground focus-visible:ring-0 focus-visible:ring-offset-0"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault();
              submit();
            }
          }}
        />
        {generating ? (
          <Button variant="default" size="icon" className="h-8 w-8 rounded-full" aria-label={t("chat.stop")} onClick={onStop}><Square size={12} fill="currentColor" /></Button>
        ) : (
          <Button size="icon" className="h-8 w-8 rounded-full" aria-label={t("chat.send")} disabled={!canSend} title={disabled ? (disabledReason ?? undefined) : undefined} onClick={submit}><ArrowUp size={16} /></Button>
        )}
      </div>
      <p className="mt-1.5 text-center text-[11px] text-muted-foreground">{disabledReason ?? t("chat.hint")}</p>
    </div>
  );
}
