import type { ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "default" | "outline" | "ghost" | "danger" | "subtle";
  size?: "sm" | "default" | "icon";
};

const variants = {
  default: "bg-primary text-primary-foreground hover:bg-primary/85",
  outline: "border bg-transparent hover:bg-muted",
  ghost: "text-muted-foreground hover:bg-muted hover:text-foreground",
  subtle: "bg-muted text-foreground hover:bg-muted/70",
  danger: "bg-destructive text-background hover:bg-destructive/85",
};

const sizes = { sm: "h-7 px-2.5 text-xs", default: "h-8 px-3 text-[13px]", icon: "h-8 w-8" };

export function Button({ className, variant = "default", size = "default", type = "button", ...props }: ButtonProps) {
  return (
    <button
      type={type}
      className={cn(
        "inline-flex shrink-0 items-center justify-center gap-1.5 whitespace-nowrap rounded-md font-medium transition-colors disabled:pointer-events-none disabled:opacity-45",
        variants[variant],
        sizes[size],
        className,
      )}
      {...props}
    />
  );
}
