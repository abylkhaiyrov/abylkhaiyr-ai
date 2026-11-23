import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { useApiError } from "@/hooks/use-api-error";
import { useRole } from "@/hooks/use-auth";
import { useRuntime } from "@/hooks/use-runtime";
import { useStudioEvents } from "@/hooks/use-studio-events";
import { ApiError } from "@/lib/http";
import { useUiStore } from "@/store/ui-store";
import { cn } from "@/lib/utils";

export function ConnectionStatus({ error, retry }: { error: Error | null; retry: () => void }) {
  const { t } = useTranslation();
  const message = useApiError();
  const { reconnect } = useStudioEvents();
  const socket = useUiStore((state) => state.socketStatus);
  const runtime = useRuntime().data;
  const stale = error !== null || socket !== "connected";
  const admin = useRole() === "ADMIN";
  if (!admin && !stale) return null;

  return (
    <div className={cn("shrink-0 border-b px-3 py-1.5 text-xs hairline", stale && "bg-warning/10")}>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
        <span role="status" className="inline-flex items-center gap-1.5"><span aria-hidden className={cn("h-1.5 w-1.5 rounded-full", stale ? "bg-warning" : "bg-success")} />{t(error ? "connection.apiUnavailable" : "connection.apiReady")} · {t(`connection.${socket}`)}</span>
        {admin ? <span>{runtime?.models.loaded?.display_name ?? t("connection.noModel")}</span> : null}
        {admin ? <span>{t(runtime ? runtime.pipeline?.running || runtime.compute.holder?.stage ? "connection.training" : "connection.idle" : "connection.unknownTraining")}</span> : null}
        {stale ? <span className="text-warning">{t("connection.stale")}</span> : null}
        {stale ? <Button size="sm" variant="outline" onClick={error ? retry : reconnect}>{t("connection.retry")}</Button> : null}
      </div>
      {stale ? (
        <details className="mt-1 text-muted-foreground">
          <summary className="cursor-pointer">{t("connection.details")}</summary>
          <p>{error ? message(error) : t(`connection.${socket}`)} {error instanceof ApiError && error.status > 0 ? `HTTP ${error.status}` : null}</p>
        </details>
      ) : null}
    </div>
  );
}
