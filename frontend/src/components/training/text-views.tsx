import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { EmptyState, ErrorState, LoadingState } from "@/components/data-state";
import { api, type RunDetail } from "@/lib/api";

export function ConfigView({ run }: { run: RunDetail }) {
  const { t } = useTranslation();
  if (run.config === null) return <EmptyState title={t("training.noConfig")} />;
  return (
    <div className="rounded-xl border">
      <p className="border-b px-3 py-2 font-mono text-xs text-muted-foreground">{run.config.path}</p>
      <pre className="max-h-[60vh] overflow-auto p-3 font-mono text-xs leading-relaxed">{run.config.text}</pre>
    </div>
  );
}

export function LogView({ name, live }: { name: string; live: boolean }) {
  const { t } = useTranslation();
  const query = useQuery({
    queryKey: ["run-logs", name],
    queryFn: () => api.runLogs(name, 300),
    refetchInterval: live ? 15_000 : false,
    staleTime: live ? 5_000 : 60_000,
  });
  if (query.isPending) return <LoadingState />;
  if (query.isError) return <ErrorState error={query.error} retry={() => void query.refetch()} />;
  if (query.data.length === 0) return <EmptyState title={t("training.noLogs")} />;
  return (
    <pre className="max-h-[60vh] overflow-auto rounded-xl border bg-muted/40 p-3 font-mono text-xs leading-relaxed">
      {query.data.join("\n")}
    </pre>
  );
}
