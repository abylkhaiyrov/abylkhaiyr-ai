import { useMemo, useState } from "react";
import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { EmptyState, ErrorState, LoadingState } from "@/components/data-state";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/select";
import { Checkbox } from "@/components/training/checkbox";
import { useApiError } from "@/hooks/use-api-error";
import { useFormat } from "@/hooks/use-format";
import { api, type CatalogModel, type CheckpointStatus } from "@/lib/api";
import { nextOffset } from "@/lib/history";
import { useToastStore } from "@/store/toast-store";

const STATUSES: CheckpointStatus[] = ["production", "validated", "candidate", "experimental", "rejected"];
const PAGE = 30;

export function CatalogTable() {
  const { t } = useTranslation();
  const format = useFormat();
  const client = useQueryClient();
  const push = useToastStore((state) => state.push);
  const describe = useApiError();
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState<CheckpointStatus | "">("");
  const query = useInfiniteQuery({
    queryKey: ["model-catalog", "table", search, status],
    queryFn: ({ pageParam }) => api.modelCatalog({ q: search.trim() || undefined, status: status || undefined, limit: PAGE, offset: pageParam }),
    initialPageParam: 0,
    getNextPageParam: (_last, pages) => nextOffset(pages),
    staleTime: 30_000,
  });
  const allowed = useQuery({ queryKey: ["model-catalog-allowed"], queryFn: api.allowedModels, staleTime: 30_000 });
  const save = useMutation({
    mutationFn: api.setAllowedModels,
    onSuccess: (next) => {
      client.setQueryData(["model-catalog-allowed"], next);
      void client.invalidateQueries({ queryKey: ["model-catalog"] });
      void client.invalidateQueries({ queryKey: ["user-models"] });
    },
    onError: (error: Error) => push("error", describe(error)),
  });
  const models = useMemo(() => query.data?.pages.flatMap((page) => page.items) ?? [], [query.data]);
  const total = query.data?.pages.at(-1)?.total ?? 0;
  const ids = allowed.data?.ids ?? [];
  const toggle = (model: CatalogModel) => save.mutate(ids.includes(model.id) ? ids.filter((item) => item !== model.id) : [...ids, model.id]);
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-end gap-3">
        <label className="grid min-w-56 flex-1 gap-1 text-xs font-medium">{t("history.search")}<input type="search" className="field" value={search} onChange={(event) => setSearch(event.target.value)} /></label>
        <label className="grid min-w-44 gap-1 text-xs font-medium">{t("models.quality")}
          <Select value={status} onChange={(event) => setStatus(event.target.value as CheckpointStatus | "")}><option value="">{t("evaluation.all")}</option>{STATUSES.map((value) => <option key={value} value={value}>{t(`checkpointStatus.${value}`)}</option>)}</Select>
        </label>
      </div>
      {query.isPending ? <LoadingState /> : query.isError ? <ErrorState error={query.error} retry={() => void query.refetch()} /> : models.length === 0 ? <EmptyState title={t("models.empty")} /> : (
        <>
          <p className="text-xs text-muted-foreground" aria-live="polite">{t("history.shown", { shown: models.length, total })}</p>
          <div className="overflow-x-auto border-y hairline">
            <table className="data-table">
              <thead><tr><th>{t("models.model")}</th><th>{t("models.version")}</th><th>{t("models.parameters")}</th><th>{t("models.quality")}</th><th>{t("models.capabilities")}</th><th>{t("models.checkpoints")}</th><th>{t("models.visibleToUsers")}</th></tr></thead>
              <tbody>
                {models.map((model) => {
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
                          <Checkbox checked={model.allowed} disabled={rejected || save.isPending || allowed.isPending} onChange={() => toggle(model)} label={t("models.visibleToUsers")} />
                          {rejected ? t("models.rejectedHint") : !model.available ? t("models.unavailableHint") : null}
                        </label>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {query.hasNextPage ? <Button variant="outline" className="justify-self-center" disabled={query.isFetchingNextPage} onClick={() => void query.fetchNextPage()}>{t("common.showMore")}</Button> : null}
        </>
      )}
    </div>
  );
}
