import { AlertTriangle, BadgeCheck } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Panel } from "@/components/data-state";
import type { ModelRole, RuntimeState } from "@/lib/api";

const ROLES: ModelRole[] = ["production", "best_evaluated", "latest_trained", "loaded"];

export function ModelRolesCard({ runtime }: { runtime: RuntimeState }) {
  const { t } = useTranslation();
  const production = runtime.production;
  return (
    <Panel title={t("roles.title")}>
      <ul className="grid gap-3">
        {ROLES.map((role) => {
          const model = runtime.models[role];
          return (
            <li key={role} className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="text-xs text-muted-foreground">{t(`role.${role}`)}</p>
                <p className="truncate font-medium" title={model?.id}>{model?.display_name ?? t(role === "production" ? "dashboard.noProduction" : "roles.none")}</p>
              </div>
              <div className="flex shrink-0 flex-col items-end gap-1 text-xs">
                {model?.stage ? <span className="rounded bg-muted px-1.5 py-0.5 uppercase">{t(`modelStage.${model.stage}`, { defaultValue: model.stage })}</span> : null}
                {model?.verified ? <span className="inline-flex items-center gap-1 text-success"><BadgeCheck size={12} />{t("roles.verified")}</span> : null}
              </div>
            </li>
          );
        })}
      </ul>
      {production && !production.ready ? (
        <div role="note" className="mt-4 flex gap-2 text-xs text-warning">
          <AlertTriangle size={14} className="mt-0.5 shrink-0" />
          <div>
            <p className="font-medium">{t("roles.notReady")}</p>
            {production.limitations.length > 0 ? <ul className="mt-1 list-disc pl-4 text-foreground/80">{production.limitations.map((item) => <li key={item}>{item}</li>)}</ul> : null}
          </div>
        </div>
      ) : null}
    </Panel>
  );
}
