import { useMemo, useState } from "react";
import { useInfiniteQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { EmptyState, ErrorState, LoadingState, PageHeader, Panel } from "@/components/data-state";
import { CatalogTable } from "@/components/models/catalog-table";
import { CheckpointTable } from "@/components/models/checkpoint-table";
import { ProfilesTable } from "@/components/models/profiles-table";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/select";
import { Tabs } from "@/components/ui/tabs";
import { useApiError } from "@/hooks/use-api-error";
import { useInferenceProfiles, useModelCatalog, useModelRegistry } from "@/hooks/use-runtime";
import { api, type CheckpointStatus, type ModelRole } from "@/lib/api";
import { nextOffset } from "@/lib/history";
import { useToastStore } from "@/store/toast-store";
import { Query } from "@/pages/resource-pages";

const ROLES: ModelRole[] = ["production", "best_evaluated", "latest_trained", "loaded"];
const STATUSES: CheckpointStatus[] = ["production", "validated", "candidate", "experimental", "rejected"];
const PAGE = 50;

export function ModelsPage() {
  const { t } = useTranslation();
  const client = useQueryClient();
  const push = useToastStore((state) => state.push);
  const describe = useApiError();
  const registry = useModelRegistry();
  const catalog = useModelCatalog();
  const profiles = useInferenceProfiles();
  const [view, setView] = useState<"catalog" | "profiles" | "checkpoints">("catalog");
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState<CheckpointStatus | "">("");
  const checkpoints = useInfiniteQuery({
    queryKey: ["checkpoints", search, status],
    queryFn: ({ pageParam }) => api.modelCheckpoints({ q: search.trim() || undefined, status: status || undefined, limit: PAGE, offset: pageParam }),
    initialPageParam: 0,
    getNextPageParam: (_last, pages) => nextOffset(pages),
    enabled: view === "checkpoints",
    staleTime: 30_000,
  });
  const models = useMemo(() => checkpoints.data?.pages.flatMap((page) => page.items) ?? [], [checkpoints.data]);
  const total = checkpoints.data?.pages.at(-1)?.total ?? 0;
  const settle = () => {
    void client.invalidateQueries({ queryKey: ["models"] });
    void client.invalidateQueries({ queryKey: ["checkpoints"] });
  };
  const fail = (error: Error) => push("error", describe(error));
  const load = useMutation({ mutationFn: (id: string) => api.switchModel(id), onSuccess: settle, onError: fail });
  const unload = useMutation({ mutationFn: api.unloadModel, onSuccess: settle, onError: fail });
  return (
    <div className="grid gap-4">
      <PageHeader title={t("models.title")} subtitle={t("models.subtitle")} actions={<Tabs value={view} onChange={setView} items={[{ value: "catalog", label: t("models.catalog") }, { value: "profiles", label: t("models.profiles") }, { value: "checkpoints", label: t("models.checkpointRegistry") }]} />} />
      {view === "catalog" ? <Query query={catalog}>{(entries) => <CatalogTable catalog={entries} />}</Query> : null}
      {view === "profiles" ? <Query query={profiles}>{(items) => <ProfilesTable profiles={items} />}</Query> : null}
      {view === "checkpoints" ? (
        <>
          <Query query={registry}>
            {(data) => (
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                {ROLES.map((role) => {
                  const model = data.models.find((item) => item.id === data.roles[role]);
                  return (
                    <Panel key={role}>
                      <p className="label-caps">{t(`role.${role}`)}</p>
                      <p className="mt-1 truncate text-base font-semibold" title={model?.id}>{model?.display_name ?? t(role === "production" ? "dashboard.noProduction" : "roles.none")}</p>
                      <p className="mt-0.5 text-xs text-muted-foreground">{t(`roleHelp.${role}`)}</p>
                    </Panel>
                  );
                })}
              </div>
            )}
          </Query>
          <div className="flex flex-wrap items-end gap-3">
            <label className="grid min-w-56 flex-1 gap-1 text-xs font-medium">{t("history.search")}<input type="search" className="field" value={search} onChange={(event) => setSearch(event.target.value)} /></label>
            <label className="grid min-w-44 gap-1 text-xs font-medium">{t("models.quality")}
              <Select value={status} onChange={(event) => setStatus(event.target.value as CheckpointStatus | "")}><option value="">{t("evaluation.all")}</option>{STATUSES.map((value) => <option key={value} value={value}>{t(`checkpointStatus.${value}`)}</option>)}</Select>
            </label>
          </div>
          {checkpoints.isPending ? <LoadingState /> : checkpoints.isError ? <ErrorState error={checkpoints.error} retry={() => void checkpoints.refetch()} /> : models.length === 0 ? <EmptyState title={t("models.empty")} /> : (
            <>
              <p className="text-xs text-muted-foreground" aria-live="polite">{t("history.shown", { shown: models.length, total })}</p>
              <CheckpointTable models={models} busy={load.isPending || unload.isPending} onLoad={(id) => load.mutate(id)} onUnload={() => unload.mutate()} />
              {checkpoints.hasNextPage ? <Button variant="outline" className="justify-self-center" disabled={checkpoints.isFetchingNextPage} onClick={() => void checkpoints.fetchNextPage()}>{t("common.showMore")}</Button> : null}
            </>
          )}
        </>
      ) : null}
    </div>
  );
}
