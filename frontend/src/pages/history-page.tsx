import { useMemo, useState } from "react";
import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { EmptyState, ErrorState, LoadingState, PageHeader } from "@/components/data-state";
import { RunCompare } from "@/components/history/run-compare";
import { RunDetailPanel } from "@/components/history/run-detail";
import { RunList } from "@/components/history/run-list";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/select";
import { api, type RunStatus } from "@/lib/api";
import { EMPTY_FILTERS, nextOffset, STAGES, STATUSES, toggleRun, toQuery, type HistoryFilters } from "@/lib/history";

export function HistoryPage() {
  const { t } = useTranslation();
  const [filters, setFilters] = useState<HistoryFilters>(EMPTY_FILTERS);
  const [openId, setOpenId] = useState<string | null>(null);
  const [compared, setCompared] = useState<string[]>([]);
  const query = useInfiniteQuery({
    queryKey: ["catalog-runs", "history", filters],
    queryFn: ({ pageParam }) => api.catalogRuns(toQuery(filters, pageParam)),
    initialPageParam: 0,
    getNextPageParam: (_last, pages) => nextOffset(pages),
    staleTime: 30_000,
  });
  const runs = useMemo(() => query.data?.pages.flatMap((page) => page.items) ?? [], [query.data]);
  const total = query.data?.pages.at(-1)?.total ?? 0;
  const known = useQuery({ queryKey: ["catalog-runs", "all-versions"], queryFn: () => api.catalogRuns({ limit: 200 }), staleTime: 60_000 });
  const versions = useMemo(() => [...new Set((known.data?.items ?? []).map((item) => item.version).filter((value): value is string => value !== null))].sort(), [known.data]);
  const comparedRuns = runs.filter((run) => compared.includes(run.id));
  const patch = (next: Partial<HistoryFilters>) => setFilters((previous) => ({ ...previous, ...next }));
  const filtered = JSON.stringify(filters) !== JSON.stringify(EMPTY_FILTERS);
  return (
    <div className="grid gap-5">
      <PageHeader title={t("history.title")} subtitle={t("history.subtitle")} />
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-[minmax(0,2fr)_repeat(4,minmax(0,1fr))_auto]">
        <label className="grid gap-1 text-xs font-medium">{t("history.search")}<input type="search" className="field" value={filters.q} onChange={(event) => patch({ q: event.target.value })} /></label>
        <label className="grid gap-1 text-xs font-medium">{t("processes.status")}
          <Select value={filters.status} onChange={(event) => patch({ status: event.target.value as RunStatus | "" })}><option value="">{t("evaluation.all")}</option>{STATUSES.map((value) => <option key={value} value={value}>{t(`status.${value}`)}</option>)}</Select>
        </label>
        <label className="grid gap-1 text-xs font-medium">{t("history.type")}
          <Select value={filters.stage} onChange={(event) => patch({ stage: event.target.value })}><option value="">{t("evaluation.all")}</option>{STAGES.map((value) => <option key={value} value={value}>{value === "pipeline" ? t("processes.pipeline") : t(`modelStage.${value}`)}</option>)}</Select>
        </label>
        <label className="grid gap-1 text-xs font-medium">{t("models.version")}
          <Select value={filters.version} onChange={(event) => patch({ version: event.target.value })}><option value="">{t("evaluation.all")}</option>{versions.map((value) => <option key={value} value={value}>{value}</option>)}</Select>
        </label>
        <label className="grid gap-1 text-xs font-medium">{t("history.sort")}
          <Select value={filters.sort} onChange={(event) => patch({ sort: event.target.value as HistoryFilters["sort"] })}><option value="newest">{t("history.newest")}</option><option value="oldest">{t("history.oldest")}</option></Select>
        </label>
        {filtered ? <Button variant="ghost" size="sm" className="self-end" onClick={() => setFilters(EMPTY_FILTERS)}>{t("common.reset")}</Button> : null}
      </div>
      {query.isPending ? <LoadingState /> : query.isError ? <ErrorState error={query.error} retry={() => void query.refetch()} /> : runs.length === 0 ? (
        <EmptyState title={t(filtered ? "history.noMatches" : "history.empty")} description={filtered ? undefined : t("history.emptyText")} />
      ) : (
        <>
          <p className="text-xs text-muted-foreground" aria-live="polite">{t("history.shown", { shown: runs.length, total })}{comparedRuns.length > 0 ? ` · ${t("evalCenter.selected", { count: comparedRuns.length })}` : ""}</p>
          <div className="grid items-start gap-8 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
            <div className="grid gap-6">
              <RunList runs={runs} total={total} openId={openId} compared={compared} onOpen={setOpenId} onCompare={(id) => setCompared((previous) => toggleRun(previous, id))} onEnd={() => query.hasNextPage && !query.isFetchingNextPage && void query.fetchNextPage()} />
              {comparedRuns.length >= 2 ? <RunCompare runs={comparedRuns} /> : compared.length === 1 ? <p className="text-xs text-muted-foreground">{t("history.pickMore")}</p> : null}
            </div>
            {openId ? <RunDetailPanel id={openId} onClose={() => setOpenId(null)} /> : <EmptyState title={t("history.selectRun")} />}
          </div>
        </>
      )}
    </div>
  );
}
