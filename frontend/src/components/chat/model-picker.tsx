import { useEffect, useRef, useState } from "react";
import { BadgeCheck, ChevronDown } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import type { Model, ModelRegistry } from "@/lib/api";
import { cn } from "@/lib/utils";

type ModelPickerProps = {
  registry: ModelRegistry | undefined;
  selected: Model | null;
  locked: boolean;
  onSelect: (model: Model) => void;
};

function reason(model: Model, t: (key: string) => string): string | null {
  if (!model.compatible) return t(`modelIssue.${model.compatibility_issues[0] ?? "incompatible_model"}`);
  if (!model.available && model.availability_reason) return t(`modelIssue.${model.availability_reason.split(" ")[0]}`);
  return null;
}

export function ModelPicker({ registry, selected, locked, onSelect }: ModelPickerProps) {
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
  const models = registry?.models ?? [];
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
            const blocked = reason(model, t);
            return (
              <li key={model.id}>
                <button
                  type="button"
                  role="option"
                  aria-selected={selected?.id === model.id}
                  disabled={blocked !== null}
                  onClick={() => { onSelect(model); setOpen(false); }}
                  className={cn("flex w-full flex-col gap-0.5 rounded-lg px-2.5 py-2 text-left hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50", selected?.id === model.id && "bg-primary/10")}
                >
                  <span className="flex items-center gap-2">
                    <span className="truncate text-sm font-medium">{model.display_name}</span>
                    <span className="rounded bg-muted px-1 text-[10px] uppercase">{t(`modelStage.${model.stage}`, { defaultValue: model.stage })}</span>
                    {model.verified ? <BadgeCheck size={13} className="text-success" aria-label={t("roles.verified")} /> : null}
                  </span>
                  <span className="flex flex-wrap gap-1 text-[11px] text-muted-foreground">
                    {model.roles.map((role) => <span key={role} className="rounded border px-1">{t(`role.${role}`)}</span>)}
                    {blocked ? <span className="text-destructive">{blocked}</span> : null}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      ) : null}
    </div>
  );
}
