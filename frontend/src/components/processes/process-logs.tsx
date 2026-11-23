import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { LoadingState } from "@/components/data-state";
import { useFormat } from "@/hooks/use-format";
import { api, type Process } from "@/lib/api";

export function ProcessLogs({ item }: { item: Process }) {
  const { t } = useTranslation();
  const format = useFormat();
  const managed = item.managed !== false;
  const events = useQuery({ queryKey: ["process-events", item.id], queryFn: () => api.processEvents(item.id), enabled: managed, retry: false, staleTime: 15_000 });
  return (
    <div className="grid gap-2 py-2 text-xs">
      <dl className="grid gap-x-6 gap-y-1 sm:grid-cols-2">
        <div><dt className="text-muted-foreground">{t("processes.command")}</dt><dd className="break-all font-mono">{item.command}</dd></div>
        {item.output_dir ? <div><dt className="text-muted-foreground">{t("processes.output")}</dt><dd className="break-all font-mono">{item.output_dir}</dd></div> : null}
      </dl>
      {!managed ? <p className="text-muted-foreground">{t("processes.noLogs")}</p> : events.isPending ? <LoadingState /> : (events.data?.length ?? 0) === 0 ? <p className="text-muted-foreground">{t("processes.noLogs")}</p> : (
        <ul className="max-h-48 divide-y overflow-auto border-y font-mono hairline">
          {events.data?.map((event, index) => <li key={index} className="grid gap-2 py-1 sm:grid-cols-[11rem_minmax(0,1fr)]"><span className="text-muted-foreground">{format.dateTime(event.created_at)}</span><span className={event.level === "error" ? "text-destructive" : undefined}>{event.message}</span></li>)}
        </ul>
      )}
    </div>
  );
}
