import { useRef, type KeyboardEvent } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useTranslation } from "react-i18next";
import { StatusBadge } from "@/components/status-badge";
import { Checkbox } from "@/components/training/checkbox";
import { useFormat } from "@/hooks/use-format";
import type { RunEntry } from "@/lib/api";
import { cn } from "@/lib/utils";

const ROW_HEIGHT = 64;

type Props = {
  runs: RunEntry[];
  total: number;
  openId: string | null;
  compared: string[];
  onOpen: (id: string) => void;
  onCompare: (id: string) => void;
  onEnd: () => void;
};

export function RunList({ runs, total, openId, compared, onOpen, onCompare, onEnd }: Props) {
  const { t } = useTranslation();
  const format = useFormat();
  const parent = useRef<HTMLDivElement>(null);
  const virtualizer = useVirtualizer({ count: runs.length, getScrollElement: () => parent.current, estimateSize: () => ROW_HEIGHT, overscan: 6 });
  const move = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    const delta = event.key === "ArrowDown" ? 1 : event.key === "ArrowUp" ? -1 : 0;
    if (delta === 0) return;
    event.preventDefault();
    const target = Math.min(Math.max(index + delta, 0), runs.length - 1);
    virtualizer.scrollToIndex(target);
    requestAnimationFrame(() => parent.current?.querySelector<HTMLButtonElement>(`[data-row="${target}"]`)?.focus());
  };
  const onScroll = () => {
    const element = parent.current;
    if (element && element.scrollHeight - element.scrollTop - element.clientHeight < ROW_HEIGHT * 4) onEnd();
  };
  return (
    <div ref={parent} onScroll={onScroll} className="max-h-[68vh] min-h-48 overflow-y-auto border-y hairline" role="list" aria-label={t("history.list")} aria-rowcount={total}>
      <div style={{ height: virtualizer.getTotalSize() }} className="relative w-full">
        {virtualizer.getVirtualItems().map((virtual) => {
          const run = runs[virtual.index];
          return (
            <div key={run.id} role="listitem" aria-posinset={virtual.index + 1} aria-setsize={total} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: virtual.size, transform: `translateY(${virtual.start}px)` }} className={cn("flex items-center gap-2 border-b border-border/60 px-2", run.id === openId && "bg-muted")}>
              <Checkbox checked={compared.includes(run.id)} label={t("history.compareRun", { name: run.name })} onChange={() => onCompare(run.id)} />
              <button type="button" data-row={virtual.index} onKeyDown={(event) => move(event, virtual.index)} onClick={() => onOpen(run.id)} aria-current={run.id === openId} className="grid min-w-0 flex-1 gap-0.5 rounded py-1 text-left hover:bg-muted/60">
                <span className="flex items-center gap-2"><span className="truncate text-[13px] font-medium" title={run.name}>{run.name}</span><StatusBadge value={run.status} /></span>
                <span className="flex flex-wrap gap-x-3 text-xs text-muted-foreground">
                  <span>{run.kind === "pipeline" ? t("processes.pipeline") : t(`modelStage.${run.stage}`, { defaultValue: run.stage })}</span>
                  {run.version ? <span>{run.version}</span> : null}
                  <span>{format.dateTime(run.updated_at)}</span>
                  {run.step !== null ? <span>{t("history.steps", { value: format.number(run.step, 0) })}</span> : null}
                  {run.validation_loss !== null ? <span>{t("metric.validation_loss")} {format.number(run.validation_loss, 3)}</span> : null}
                </span>
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
