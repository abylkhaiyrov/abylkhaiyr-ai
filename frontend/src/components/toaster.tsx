import { CheckCircle2, Info, XCircle } from "lucide-react";
import { useToastStore } from "@/store/toast-store";
import { cn } from "@/lib/utils";

const icons = { success: CheckCircle2, error: XCircle, info: Info };
const tones = { success: "text-success", error: "text-destructive", info: "text-info" };

export function Toaster() {
  const toasts = useToastStore((state) => state.toasts);
  const dismiss = useToastStore((state) => state.dismiss);
  return (
    <div aria-live="polite" className="fixed bottom-4 right-4 z-[60] grid w-[min(22rem,calc(100vw-2rem))] gap-2">
      {toasts.map((toast) => {
        const Icon = icons[toast.kind];
        return (
          <button key={toast.id} type="button" onClick={() => dismiss(toast.id)} className="flex items-start gap-2 rounded-lg border bg-card p-3 text-left text-sm shadow-lg animate-rise">
            <Icon size={16} className={cn("mt-0.5 shrink-0", tones[toast.kind])} />
            <span className="min-w-0 break-words">{toast.message}</span>
          </button>
        );
      })}
    </div>
  );
}
