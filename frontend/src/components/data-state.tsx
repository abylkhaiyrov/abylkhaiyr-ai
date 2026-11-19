import { AlertCircle, Inbox } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";

export function LoadingState() {
  const { t } = useTranslation();
  return <div role="status" aria-busy="true" aria-label={t("common.loading")} className="grid gap-3"><div className="h-28 animate-pulse rounded-xl bg-muted" /><div className="h-48 animate-pulse rounded-xl bg-muted" /></div>;
}

export function EmptyState({ title, description }: { title: string; description: string }) {
  return <div className="grid place-items-center rounded-xl border border-dashed p-10 text-center"><Inbox size={26} className="mb-3 text-muted-foreground" /><h2 className="font-medium">{title}</h2><p className="mt-1 max-w-md text-sm text-muted-foreground">{description}</p></div>;
}

export function ErrorState({ error, retry }: { error: Error; retry: () => void }) {
  const { t } = useTranslation();
  return <div role="alert" className="flex items-start gap-3 rounded-xl border border-destructive/30 bg-destructive/5 p-5"><AlertCircle className="mt-0.5 shrink-0 text-destructive" size={18} /><div><p className="font-medium">{t("errors.load")}</p><p className="mt-1 text-sm text-muted-foreground">{error.message}</p><Button variant="outline" className="mt-3 h-8" onClick={retry}>{t("common.retry")}</Button></div></div>;
}
