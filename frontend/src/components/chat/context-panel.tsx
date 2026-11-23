import { useEffect, useRef, useState } from "react";
import { Gauge, SquarePen } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { ProgressBar } from "@/components/ui/progress-bar";
import { Select } from "@/components/ui/select";
import { useFormat } from "@/hooks/use-format";

const WINDOWS = [0, 2, 4, 8, 16];

type ContextPanelProps = {
  used: number | null;
  limit: number | null;
  messageCount: number;
  window: number;
  disabled: boolean;
  onWindow: (value: number) => void;
  onSystemPrompt: () => void;
  onClear: () => void;
};

export function ContextPanel({ used, limit, messageCount, window: windowSize, disabled, onWindow, onSystemPrompt, onClear }: ContextPanelProps) {
  const { t } = useTranslation();
  const format = useFormat();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (event: MouseEvent) => !ref.current?.contains(event.target as Node) && setOpen(false);
    const key = (event: KeyboardEvent) => event.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", key);
    };
  }, [open]);
  const percent = used !== null && limit ? Math.min(100, (used / limit) * 100) : null;
  return (
    <div ref={ref} className="relative">
      <Button variant="ghost" size="sm" aria-haspopup="dialog" aria-expanded={open} onClick={() => setOpen((value) => !value)}>
        <Gauge size={14} />
        <span className="tabular-nums">{percent === null ? t("chat.contextShort") : `${format.number(percent, 0)}%`}</span>
      </Button>
      {open ? (
        <div role="dialog" aria-label={t("chat.contextTitle")} className="absolute right-0 top-full z-30 mt-1 grid w-72 max-w-[88vw] gap-3 rounded-xl border bg-background p-3 shadow-xl animate-rise">
          <p className="text-[13px] font-semibold">{t("chat.contextTitle")}</p>
          <div className="grid gap-1.5">
            <ProgressBar percent={percent} label={t("chat.contextTitle")} tone={percent !== null && percent > 85 ? "warning" : "primary"} />
            <p className="text-xs text-muted-foreground tabular-nums">{used === null || !limit ? t("chat.contextEmpty") : t("chat.context", { used: format.number(used, 0), limit: format.number(limit, 0) })} · {t("chat.messagesCount", { count: messageCount })}</p>
          </div>
          <label className="grid gap-1 text-xs font-medium">{t("chat.contextWindow")}
            <Select disabled={disabled} value={windowSize} onChange={(event) => onWindow(Number(event.target.value))}>
              {WINDOWS.map((value) => <option key={value} value={value}>{value === 0 ? t("chat.contextAll") : t("chat.contextLast", { count: value })}</option>)}
            </Select>
            <span className="font-normal text-muted-foreground">{t("chat.contextHelp")}</span>
          </label>
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant="outline" onClick={() => { setOpen(false); onSystemPrompt(); }}>{t("chat.systemPrompt")}</Button>
            <Button size="sm" variant="outline" onClick={() => { setOpen(false); onClear(); }}><SquarePen size={13} />{t("chat.clearContext")}</Button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
