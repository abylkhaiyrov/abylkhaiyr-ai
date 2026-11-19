import { cn } from "@/lib/utils";
import { useTranslation } from "react-i18next";

const styles: Record<string, string> = {
  running: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300",
  ready: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300",
  completed: "bg-sky-500/15 text-sky-700 dark:text-sky-300",
  failed: "bg-red-500/15 text-red-700 dark:text-red-300",
  stopped: "bg-amber-500/15 text-amber-700 dark:text-amber-300",
  pausing: "bg-amber-500/15 text-amber-700 dark:text-amber-300",
};

export function StatusBadge({ value }: { value: string }) {
  const { t } = useTranslation();
  return <span className={cn("inline-flex rounded-full px-2 py-0.5 text-xs font-medium", styles[value] ?? "bg-muted text-muted-foreground")}>{t(`common.${value}`, { defaultValue: value })}</span>;
}
