import { CheckCircle2, Info, XCircle } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useToastStore } from "@/store/toast-store";
import { cn } from "@/lib/utils";

const icons = { success: CheckCircle2, error: XCircle, info: Info };
const tones = { success: "text-success", error: "text-destructive", info: "text-info" };

export function Toaster() {
  const toasts = useToastStore((state) => state.toasts);
  const dismiss = useToastStore((state) => state.dismiss);
  const navigate = useNavigate();
  return (
    <div aria-live="polite" className="fixed bottom-4 right-4 z-[60] grid w-[min(22rem,calc(100vw-2rem))] gap-2">
      {toasts.map((toast) => {
        const Icon = icons[toast.kind];
        return (
          <div key={toast.id} className="flex items-start gap-2 rounded-lg border bg-background p-3 text-sm shadow-lg animate-rise">
            <Icon size={16} className={cn("mt-0.5 shrink-0", tones[toast.kind])} />
            <button type="button" className="min-w-0 flex-1 break-words text-left" onClick={() => dismiss(toast.id)}>{toast.message}</button>
            {toast.action ? (
              <button type="button" className="shrink-0 font-medium text-accent hover:underline" onClick={() => { navigate(toast.action?.href ?? "/"); dismiss(toast.id); }}>{toast.action.label}</button>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}
