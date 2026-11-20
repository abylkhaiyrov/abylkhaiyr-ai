import { cn } from "@/lib/utils";

type TabsProps<T extends string> = {
  value: T;
  onChange: (value: T) => void;
  items: { value: T; label: string }[];
  className?: string;
};

export function Tabs<T extends string>({ value, onChange, items, className }: TabsProps<T>) {
  return (
    <div role="tablist" className={cn("inline-flex max-w-full gap-1 overflow-x-auto rounded-lg bg-muted p-1", className)}>
      {items.map((item) => (
        <button
          key={item.value}
          type="button"
          role="tab"
          aria-selected={item.value === value}
          onClick={() => onChange(item.value)}
          className={cn(
            "whitespace-nowrap rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
            item.value === value ? "bg-card text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground",
          )}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}
