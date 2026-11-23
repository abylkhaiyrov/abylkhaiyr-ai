import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { EmptyState } from "@/components/data-state";
import { Checkbox } from "@/components/training/checkbox";
import { useApiError } from "@/hooks/use-api-error";
import { useFormat } from "@/hooks/use-format";
import { api, type CatalogModel } from "@/lib/api";
import { useToastStore } from "@/store/toast-store";

export function CatalogTable({ catalog }: { catalog: CatalogModel[] }) {
  const { t } = useTranslation();
  const format = useFormat();
  const client = useQueryClient();
  const push = useToastStore((state) => state.push);
  const describe = useApiError();
  const save = useMutation({
    mutationFn: api.setAllowedModels,
    onSuccess: (next) => client.setQueryData(["model-catalog"], next),
    onError: (error: Error) => push("error", describe(error)),
  });
  if (catalog.length === 0) return <EmptyState title={t("models.empty")} />;
  const toggle = (model: CatalogModel) => {
    const ids = catalog.filter((item) => (item.id === model.id ? !item.allowed : item.allowed)).map((item) => item.id);
    save.mutate(ids);
  };
  return (
    <div className="overflow-x-auto border-y hairline">
      <table className="data-table">
        <thead><tr><th>{t("models.model")}</th><th>{t("models.version")}</th><th>{t("models.parameters")}</th><th>{t("models.quality")}</th><th>{t("models.capabilities")}</th><th>{t("models.checkpoints")}</th><th>{t("models.visibleToUsers")}</th></tr></thead>
        <tbody>
          {catalog.map((model) => {
            const rejected = model.quality_status === "rejected";
            return (
              <tr key={model.id}>
                <td className="max-w-64"><p className="truncate font-medium" title={model.id}>{model.display_name}</p><p className="truncate text-xs text-muted-foreground" title={model.representative_checkpoint}>{model.id}</p></td>
                <td>{model.version ?? "—"}</td>
                <td className="tabular-nums">{format.compact(model.parameters)}</td>
                <td><span className="rounded bg-muted px-1.5 py-0.5 text-xs">{t(`checkpointStatus.${model.quality_status}`)}</span><p className="mt-1 text-xs text-muted-foreground">{t(`modelUse.${model.recommended_use}`)}</p></td>
                <td className="text-xs">{model.capabilities.map((item) => t(`capability.${item}`)).join(", ")}</td>
                <td className="tabular-nums">{model.checkpoint_count}</td>
                <td>
                  <label className="inline-flex items-center gap-2 text-xs">
                    <Checkbox checked={model.allowed} disabled={rejected || save.isPending} onChange={() => toggle(model)} label={t("models.visibleToUsers")} />
                    {rejected ? t("models.rejectedHint") : !model.available ? t("models.unavailableHint") : null}
                  </label>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
