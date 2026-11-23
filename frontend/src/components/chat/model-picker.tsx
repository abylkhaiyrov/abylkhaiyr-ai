import { useEffect, useRef, useState } from "react";
import { BadgeCheck, ChevronDown } from "lucide-react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import type { CatalogModel, Model } from "@/lib/api";
import { cn } from "@/lib/utils";

type ModelPickerProps = {
  catalog: CatalogModel[] | undefined;
  selected: Pick<Model, "model_id" | "display_name" | "stage"> | null;
  locked: boolean;
  onSelect: (model: CatalogModel) => void;
};

export function ModelPicker({ catalog, selected, locked, onSelect }: ModelPickerProps) {
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
  const models = (catalog ?? []).filter((model) => model.quality_status !== "rejected");
  return (
    <div ref={ref} className="relative">
      <Button variant="ghost" size="sm" aria-haspopup="listbox" aria-expanded={open} disabled={locked} title={locked ? t("chat.modelLocked") : undefined} onClick={() => setOpen((value) => !value)} className="max-w-[16rem] font-semibold text-foreground">
        <span className="truncate">{selected?.display_name ?? t("chat.noModel")}</span>
        {selected ? <span className="rounded bg-muted px-1 text-[10px] uppercase">{t(`modelStage.${selected.stage}`, { defaultValue: selected.stage })}</span> : null}
        <ChevronDown size={13} />
      </Button>
      {open ? (
        <ul role="listbox" aria-label={t("chat.selectModel")} className="absolute left-0 top-full z-30 mt-1 max-h-80 w-80 max-w-[85vw] overflow-y-auto rounded-xl border bg-background p-1 shadow-xl animate-rise">
          {models.length === 0 ? <li className="p-3 text-sm text-muted-foreground">{t("chat.noModels")}</li> : null}
          {models.map((model) => {
            const current = selected?.model_id === model.id;
            return (
              <li key={model.id}>
                <button
                  type="button"
                  role="option"
                  aria-selected={current}
                  disabled={!model.available}
                  onClick={() => { onSelect(model); setOpen(false); }}
                  className={cn("flex w-full flex-col gap-0.5 rounded-lg px-2.5 py-2 text-left hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50", current && "bg-primary/10")}
                >
                  <span className="flex items-center gap-2">
                    <span className="truncate text-sm font-medium">{model.display_name}</span>
                    <span className="rounded bg-muted px-1 text-[10px] uppercase">{t(`checkpointStatus.${model.quality_status}`)}</span>
                    {model.quality_status === "validated" || model.quality_status === "production" ? <BadgeCheck size={13} className="text-success" aria-label={t("roles.verified")} /> : null}
                  </span>
                  <span className="flex flex-wrap gap-x-2 text-[11px] text-muted-foreground">
                    <span>{t(`modelStage.${model.stage}`, { defaultValue: model.stage })}</span>
                    {model.parameters ? <span>{t("chat.parameters", { value: new Intl.NumberFormat(undefined, { notation: "compact" }).format(model.parameters) })}</span> : null}
                    <span>{t(`modelUse.${model.recommended_use}`)}</span>
                    {!model.available ? <span className="text-destructive">{t("chat.unavailable")}</span> : null}
                  </span>
                </button>
              </li>
            );
          })}
          <li className="border-t p-1"><Link to="/models" onClick={() => setOpen(false)} className="block rounded-lg px-2.5 py-1.5 text-xs text-muted-foreground hover:bg-muted hover:text-foreground">{t("chat.allCheckpoints")}</Link></li>
        </ul>
      ) : null}
    </div>
  );
}
