import { useTranslation } from "react-i18next";
import { StatusBadge } from "@/components/status-badge";
import { Button } from "@/components/ui/button";
import { useFormat } from "@/hooks/use-format";
import { useMetricFormat } from "@/hooks/use-metric-format";
import type { Model } from "@/lib/api";
import { modelAvailabilityKey } from "@/lib/models";

type Props = { models: Model[]; busy: boolean; onLoad: (id: string) => void; onUnload: () => void };

export function CheckpointTable({ models, busy, onLoad, onUnload }: Props) {
  const { t } = useTranslation();
  const format = useFormat();
  const metric = useMetricFormat();
  return (
    <div className="overflow-x-auto border-y hairline">
      <table className="data-table">
        <thead><tr><th>{t("models.model")}</th><th>{t("models.stage")}</th><th>{t("models.roles")}</th><th>{t("models.parameters")}</th><th>{t("models.context")}</th><th>{t("models.size")}</th><th>{t("models.status")}</th><th /></tr></thead>
        <tbody>
          {models.map((model) => {
            const issue = modelAvailabilityKey(model);
            const blocked = issue ? t(`modelIssue.${issue}`) : null;
            return (
              <tr key={model.id}>
                <td className="max-w-64"><p className="truncate font-medium" title={model.id}>{model.display_name}</p><p className="truncate text-xs text-muted-foreground" title={model.id}>{model.id}</p></td>
                <td><span className="rounded bg-muted px-1.5 py-0.5 text-xs uppercase">{t(`modelStage.${model.stage}`, { defaultValue: model.stage })}</span><p className="mt-1 text-xs text-muted-foreground">{t(`checkpointStatus.${model.status}`)}</p></td>
                <td><div className="flex flex-wrap gap-1">{model.roles.map((role) => <span key={role} className="rounded border px-1.5 py-0.5 text-xs">{t(`role.${role}`)}</span>)}</div></td>
                <td className="tabular-nums">{format.compact(model.parameters)}</td>
                <td className="tabular-nums">{format.number(model.context_length, 0)}</td>
                <td className="tabular-nums">{format.bytes(model.weights_bytes)}</td>
                <td>
                  <StatusBadge value={model.loaded ? "active" : blocked ? "failed" : "ready"} />
                  <ul className="mt-1 grid gap-0.5 text-xs text-muted-foreground">
                    {model.weights_bytes !== null ? <li>{t("modelEvidence.checkpoint")}</li> : null}
                    {model.trained ? <li>{t("modelEvidence.trained")}</li> : null}
                    {model.evaluated ? <li>{t("modelEvidence.evaluated")}</li> : null}
                    {model.quality_failed ? <li className="text-warning">{t("modelEvidence.qualityFailed")}</li> : null}
                    {!model.roles.includes("production") ? <li>{t("modelEvidence.experimental")}</li> : null}
                  </ul>
                  {blocked ? <p className="mt-1 text-xs text-destructive">{blocked}</p> : null}
                  {model.metrics?.["validation/loss"] !== undefined ? <p className="mt-1 text-xs text-muted-foreground">{metric.label("validation_loss")} {metric.value("validation_loss", model.metrics["validation/loss"])}</p> : null}
                </td>
                <td className="text-right">
                  {model.loaded ? <Button size="sm" variant="outline" disabled={busy} onClick={onUnload}>{t("models.unload")}</Button> : <Button size="sm" disabled={blocked !== null || busy} title={blocked ?? undefined} onClick={() => onLoad(model.id)}>{t("models.load")}</Button>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
