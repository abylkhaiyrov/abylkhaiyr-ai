import { AlertCircle, Inbox } from "lucide-react";
import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { useApiError } from "@/hooks/use-api-error";

export function LoadingState() {
  const { t } = useTranslation();
  return (
    <div role="status" aria-busy="true" aria-label={t("common.loading")} className="grid gap-3">
      <div className="h-24 animate-pulse rounded-xl bg-muted" />
      <div className="h-40 animate-pulse rounded-xl bg-muted" />
    </div>
  );
}

export function EmptyState({ title, description }: { title: string; description?: string }) {
  return (
    <div className="grid place-items-center rounded-xl border border-dashed p-8 text-center">
      <Inbox size={22} className="mb-2 text-muted-foreground" />
      <p className="font-medium">{title}</p>
      {description ? <p className="mt-1 max-w-md text-sm text-muted-foreground">{description}</p> : null}
    </div>
  );
}

export function ErrorState({ error, retry }: { error: Error; retry: () => void }) {
  const { t } = useTranslation();
  const message = useApiError();
  return (
    <div role="alert" className="flex items-start gap-3 rounded-xl border border-destructive/30 bg-destructive/5 p-4">
      <AlertCircle className="mt-0.5 shrink-0 text-destructive" size={18} />
      <div>
        <p className="font-medium">{t("errors.load")}</p>
        <p className="mt-1 text-sm text-muted-foreground">{message(error)}</p>
        <Button variant="outline" size="sm" className="mt-3" onClick={retry}>{t("common.retry")}</Button>
      </div>
    </div>
  );
}

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: string; actions?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-3">
      <div className="min-w-0">
        <h1 className="page-title">{title}</h1>
        {subtitle ? <p className="page-subtitle mt-0.5">{subtitle}</p> : null}
      </div>
      {actions}
    </div>
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="min-w-0">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="mt-0.5 truncate text-[15px] font-medium tabular-nums" title={typeof value === "string" ? value : undefined}>{value}</dd>
      {hint ? <p className="mt-0.5 truncate text-xs text-muted-foreground">{hint}</p> : null}
    </div>
  );
}

export function Panel({ title, actions, children, className }: { title?: string; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={className}>
      {title ? (
        <div className="mb-3 flex items-center justify-between gap-3">
          <h2 className="text-[13px] font-semibold">{title}</h2>
          {actions}
        </div>
      ) : null}
      {children}
    </section>
  );
}
