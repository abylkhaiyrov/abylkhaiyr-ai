import { useTranslation } from "react-i18next";
import { Checkbox } from "@/components/training/checkbox";
import { ProgressBar } from "@/components/ui/progress-bar";
import { StatusBadge } from "@/components/status-badge";
import { useFormat } from "@/hooks/use-format";
import type { Run } from "@/lib/api";
import { cn } from "@/lib/utils";

type RunListProps = {
  runs: Run[];
  selected: string | null;
  compared: string[];
  onSelect: (name: string) => void;
  onToggleCompare: (name: string) => void;
};

export function RunList({ runs, selected, compared, onSelect, onToggleCompare }: RunListProps) {
  const { t } = useTranslation();
  const format = useFormat();
  const reference = runs.find((run) => run.name === selected);
  return (
    <ul className="grid gap-1.5">
      {runs.map((run) => {
        const blocked = reference !== undefined && run.stage !== reference.stage && run.name !== selected;
        return (
          <li key={run.name} className={cn("flex items-start gap-2 rounded-lg border p-2.5 transition-colors", run.name === selected ? "border-primary bg-primary/5" : "hover:bg-muted/50")}>
            <Checkbox
              checked={compared.includes(run.name)}
              disabled={blocked}
              label={blocked ? t("training.compareBlocked") : t("training.compare")}
              onChange={() => onToggleCompare(run.name)}
            />
            <button type="button" className="min-w-0 flex-1 text-left" onClick={() => onSelect(run.name)}>
              <span className="flex items-center justify-between gap-2">
                <span className="truncate font-medium">{run.name}</span>
                <StatusBadge value={run.status} />
              </span>
              <span className="mt-0.5 flex items-center justify-between gap-2 text-xs text-muted-foreground">
                <span className="uppercase">{t(`modelStage.${run.stage}`)}</span>
                <span className="tabular-nums">{format.ratio(run.step, run.max_steps)}</span>
              </span>
              {run.status === "running" || (run.percent !== null && run.percent < 100) ? <ProgressBar className="mt-1.5 h-1.5" percent={run.percent} label={run.name} /> : null}
              {run.best_validation_loss !== null ? (
                <span className="mt-1 block text-xs text-muted-foreground">{t("training.bestVal")}: <span className="tabular-nums text-foreground">{format.number(run.best_validation_loss, 3)}</span></span>
              ) : null}
            </button>
          </li>
        );
      })}
    </ul>
  );
}
