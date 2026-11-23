import type { SelectHTMLAttributes } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

type SelectProps = SelectHTMLAttributes<HTMLSelectElement> & { compact?: boolean; inline?: boolean };

export function Select({ className, compact = false, inline = false, children, ...props }: SelectProps) {
  return (
    <span className={cn("relative inline-grid max-w-full", !compact && !inline && "w-full", inline && "min-w-32")}>
      <select
        className={cn(
          "select-field h-8 w-full min-w-0 truncate rounded-md border bg-background px-2.5 text-[13px] text-foreground hover:border-muted-foreground/40 disabled:cursor-not-allowed disabled:opacity-50",
          compact && "border-transparent pl-2 pr-6 text-xs text-muted-foreground hover:bg-muted",
          className,
        )}
        {...props}
      >
        {children}
      </select>
      <ChevronDown size={13} aria-hidden className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground" />
    </span>
  );
}
