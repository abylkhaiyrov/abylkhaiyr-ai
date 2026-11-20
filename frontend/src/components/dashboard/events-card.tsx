import { useTranslation } from "react-i18next";
import { Panel } from "@/components/data-state";
import { useFormat } from "@/hooks/use-format";
import type { StudioEvent } from "@/lib/api";

export function EventsCard({ events }: { events: StudioEvent[] }) {
  const { t } = useTranslation();
  const format = useFormat();
  return (
    <Panel title={t("events.title")}>
      {events.length === 0 ? (
        <p className="text-sm text-muted-foreground">{t("events.empty")}</p>
      ) : (
        <ul className="grid gap-2.5">
          {events.slice(0, 6).map((event) => (
            <li key={`${event.created_at}-${event.message}`} className="flex items-baseline justify-between gap-3 text-sm">
              <span className="min-w-0 truncate">{event.message}</span>
              <span className="shrink-0 text-xs text-muted-foreground tabular-nums">{format.dateTime(event.created_at)}</span>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}
