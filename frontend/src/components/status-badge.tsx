import { useTranslation } from "react-i18next";
import { cn } from "@/lib/utils";

const dots: Record<string, string> = {
  running: "bg-info animate-pulse",
  ready: "bg-success",
  completed: "bg-success",
  active: "bg-success",
  failed: "bg-destructive",
  interrupted: "bg-warning",
  stopped: "bg-warning",
  pausing: "bg-warning",
  paused: "bg-warning",
  busy: "bg-warning",
  skipped: "bg-muted-foreground/50",
  pending: "bg-muted-foreground/40",
  idle: "bg-muted-foreground/50",
};

export function StatusBadge({ value, className }: { value: string; className?: string }) {
  const { t } = useTranslation();
  return (
    <span className={cn("inline-flex items-center gap-1.5 whitespace-nowrap text-xs font-medium text-muted-foreground", className)}>
      <span className={cn("h-1.5 w-1.5 rounded-full", dots[value] ?? "bg-muted-foreground/50")} />
      {t(`status.${value}`, { defaultValue: value })}
    </span>
  );
}
