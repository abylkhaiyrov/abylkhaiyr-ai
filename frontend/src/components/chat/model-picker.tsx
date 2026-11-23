import { useEffect, useRef, useState } from "react";
import { AlertTriangle, BadgeCheck, ChevronDown } from "lucide-react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import type { UserModel } from "@/lib/api";
import { hasProduction } from "@/lib/profiles";
import { cn } from "@/lib/utils";

type ModelPickerProps = {
  models: UserModel[];
  selectedId: string | null;
  locked: boolean;
  onSelect: (model: UserModel) => void;
  advancedHref?: string;
};

const CAUTION = new Set(["experimental", "candidate"]);

export function ModelPicker({ models, selectedId, locked, onSelect, advancedHref }: ModelPickerProps) {
  const { t } = useTranslation();
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
  const visible = models.filter((model) => model.quality_status !== "rejected");
  const selected = visible.find((model) => model.id === selectedId) ?? null;
  return (
    <div ref={ref} className="relative">
      <Button variant="ghost" size="sm" aria-haspopup="listbox" aria-expanded={open} disabled={locked} title={locked ? t("chat.modelLocked") : undefined} onClick={() => setOpen((value) => !value)} className="max-w-[16rem] font-semibold text-foreground">
        <span className="truncate">{selected?.display_name ?? t("chat.noModel")}</span>
        {selected && CAUTION.has(selected.quality_status) ? <span className="rounded bg-warning/15 px-1 text-[10px] uppercase text-warning">{t(`checkpointStatus.${selected.quality_status}`)}</span> : null}
        <ChevronDown size={13} />
      </Button>
      {open ? (
        <div className="absolute left-0 top-full z-30 mt-1 w-80 max-w-[85vw] rounded-xl border bg-background p-1 shadow-xl animate-rise">
          {visible.length > 0 && !hasProduction(visible) ? (
            <p role="note" className="flex gap-2 px-2.5 py-2 text-xs text-warning"><AlertTriangle size={13} className="mt-0.5 shrink-0" />{t("chat.productionMissing")}</p>
          ) : null}
          <ul role="listbox" aria-label={t("chat.selectModel")} className="max-h-72 overflow-y-auto">
            {visible.length === 0 ? <li className="p-3 text-sm text-muted-foreground">{t("chat.noModels")}</li> : null}
            {visible.map((model) => (
              <li key={model.id}>
                <button
                  type="button"
                  role="option"
                  aria-selected={model.id === selectedId}
                  disabled={!model.available}
                  onClick={() => { onSelect(model); setOpen(false); }}
                  className={cn("flex w-full flex-col gap-0.5 rounded-lg px-2.5 py-2 text-left hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50", model.id === selectedId && "bg-primary/10")}
                >
                  <span className="flex items-center gap-2">
                    <span className="truncate text-sm font-medium">{model.display_name}</span>
                    <span className={cn("rounded px-1 text-[10px] uppercase", CAUTION.has(model.quality_status) ? "bg-warning/15 text-warning" : "bg-muted")}>{t(`checkpointStatus.${model.quality_status}`)}</span>
                    {model.quality_status === "validated" || model.quality_status === "production" ? <BadgeCheck size={13} className="text-success" aria-label={t("roles.verified")} /> : null}
                  </span>
                  <span className="flex flex-wrap gap-x-2 text-[11px] text-muted-foreground">
                    <span>{t("chat.family", { value: model.family })}</span>
                    {model.parameters ? <span>{t("chat.parameters", { value: new Intl.NumberFormat(undefined, { notation: "compact" }).format(model.parameters) })}</span> : null}
                    <span>{t(`modelUse.${model.recommended_use}`)}</span>
                    {!model.available ? <span className="text-destructive">{t("chat.unavailable")}</span> : null}
                  </span>
                </button>
              </li>
            ))}
          </ul>
          {advancedHref ? <div className="border-t p-1"><Link to={advancedHref} onClick={() => setOpen(false)} className="block rounded-lg px-2.5 py-1.5 text-xs text-muted-foreground hover:bg-muted hover:text-foreground">{t("chat.allCheckpoints")}</Link></div> : null}
        </div>
      ) : null}
    </div>
  );
}
