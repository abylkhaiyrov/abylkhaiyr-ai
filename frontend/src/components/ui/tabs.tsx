import { cn } from "@/lib/utils";

type TabsProps<T extends string> = {
  value: T;
  onChange: (value: T) => void;
  items: { value: T; label: string }[];
  className?: string;
};

export function Tabs<T extends string>({ value, onChange, items, className }: TabsProps<T>) {
  return (
    <div role="tablist" className={cn("flex max-w-full gap-4 overflow-x-auto border-b hairline", className)}>
      {items.map((item) => (
        <button
          key={item.value}
          type="button"
          role="tab"
          aria-selected={item.value === value}
          onClick={() => onChange(item.value)}
          className={cn(
            "-mb-px whitespace-nowrap border-b-2 px-0.5 pb-2 pt-1 text-[13px] font-medium transition-colors",
            item.value === value ? "border-foreground text-foreground" : "border-transparent text-muted-foreground hover:text-foreground",
          )}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}
