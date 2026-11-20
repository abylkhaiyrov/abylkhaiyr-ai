import { useTranslation } from "react-i18next";
import { cn } from "@/lib/utils";

const styles: Record<string, string> = {
  running: "bg-info/15 text-info",
  ready: "bg-success/15 text-success",
  completed: "bg-success/15 text-success",
  active: "bg-success/15 text-success",
  failed: "bg-destructive/15 text-destructive",
  interrupted: "bg-warning/15 text-warning",
  stopped: "bg-warning/15 text-warning",
  pausing: "bg-warning/15 text-warning",
  busy: "bg-warning/15 text-warning",
  skipped: "bg-muted text-muted-foreground",
  pending: "bg-muted text-muted-foreground",
  idle: "bg-muted text-muted-foreground",
};

export function StatusBadge({ value, className }: { value: string; className?: string }) {
  const { t } = useTranslation();
  return (
    <span className={cn("inline-flex items-center gap-1.5 whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium", styles[value] ?? "bg-muted text-muted-foreground", className)}>
      {value === "running" ? <span className="h-1.5 w-1.5 rounded-full bg-current animate-pulse" /> : null}
      {t(`status.${value}`, { defaultValue: value })}
    </span>
  );
}
