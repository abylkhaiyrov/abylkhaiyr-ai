import { cn } from "@/lib/utils";

type ProgressBarProps = {
  percent: number | null;
  indeterminate?: boolean;
  label: string;
  tone?: "primary" | "success" | "warning" | "danger";
  className?: string;
};

const tones = { primary: "bg-accent", success: "bg-success", warning: "bg-warning", danger: "bg-destructive" };

export function ProgressBar({ percent, indeterminate = false, label, tone = "primary", className }: ProgressBarProps) {
  const unknown = indeterminate || percent === null;
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={unknown ? undefined : Math.round(percent)}
      className={cn("relative h-1.5 w-full overflow-hidden rounded-full bg-muted", className)}
    >
      {unknown ? (
        <div className={cn("absolute inset-y-0 w-1/4 rounded-full animate-indeterminate", tones[tone])} />
      ) : (
        <div className={cn("h-full rounded-full transition-[width] duration-500", tones[tone])} style={{ width: `${Math.min(100, Math.max(0, percent))}%` }} />
      )}
    </div>
  );
}
