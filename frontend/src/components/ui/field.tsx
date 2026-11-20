import { useId, useState, type InputHTMLAttributes } from "react";
import { Eye, EyeOff } from "lucide-react";
import { useTranslation } from "react-i18next";
import { cn } from "@/lib/utils";

type FieldProps = Omit<InputHTMLAttributes<HTMLInputElement>, "id"> & { label: string; error?: string | null; hint?: string };

export function TextField({ label, error, hint, className, ...props }: FieldProps) {
  const id = useId();
  return (
    <div className="grid gap-1.5">
      <label htmlFor={id} className="text-[13px] font-medium">{label}</label>
      <input id={id} aria-invalid={error ? true : undefined} aria-describedby={error || hint ? `${id}-note` : undefined} className={cn("field h-9", error && "border-destructive", className)} {...props} />
      {error || hint ? <p id={`${id}-note`} role={error ? "alert" : undefined} className={cn("text-xs", error ? "text-destructive" : "text-muted-foreground")}>{error ?? hint}</p> : null}
    </div>
  );
}

export function PasswordField({ label, error, hint, className, ...props }: FieldProps) {
  const { t } = useTranslation();
  const id = useId();
  const [visible, setVisible] = useState(false);
  return (
    <div className="grid gap-1.5">
      <label htmlFor={id} className="text-[13px] font-medium">{label}</label>
      <div className="relative">
        <input id={id} type={visible ? "text" : "password"} aria-invalid={error ? true : undefined} aria-describedby={error || hint ? `${id}-note` : undefined} className={cn("field h-9 pr-9", error && "border-destructive", className)} {...props} />
        <button type="button" aria-label={t(visible ? "auth.hidePassword" : "auth.showPassword")} aria-pressed={visible} onClick={() => setVisible((value) => !value)} className="absolute right-1 top-1 grid h-7 w-7 place-items-center rounded text-muted-foreground hover:bg-muted hover:text-foreground">
          {visible ? <EyeOff size={15} /> : <Eye size={15} />}
        </button>
      </div>
      {error || hint ? <p id={`${id}-note`} role={error ? "alert" : undefined} className={cn("text-xs", error ? "text-destructive" : "text-muted-foreground")}>{error ?? hint}</p> : null}
    </div>
  );
}
