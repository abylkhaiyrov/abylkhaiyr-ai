import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { ComparePanel } from "@/components/evaluation/compare-panel";
import { GateList } from "@/components/evaluation/gate-list";
import { MetricSection } from "@/components/evaluation/metric-section";
import { EmptyState, ErrorState, LoadingState, PageHeader, Panel, Stat } from "@/components/data-state";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/select";
import { Tabs } from "@/components/ui/tabs";
import { Checkbox } from "@/components/training/checkbox";
import { useFormat } from "@/hooks/use-format";
import { api, type EvaluationGroup, type MetricDefinition } from "@/lib/api";
import { checkpointLabel, DEFAULT_SELECTION, rowId, SECTIONS, sectionCoverage, toggle, type SectionKey } from "@/lib/evaluation";

type View = "overview" | SectionKey | "compare";
const PAGE = 30;

function Overview({ group, catalog, selected, onToggle, others }: { group: EvaluationGroup; catalog: MetricDefinition[]; selected: string[]; onToggle: (id: string) => void; others: number }) {
  const { t } = useTranslation();
  const format = useFormat();
  const missing = t("evalCenter.notRecorded");
  return (
    <div className="grid gap-8">
      <dl className="grid grid-cols-2 gap-x-6 gap-y-4 border-b hairline pb-6 md:grid-cols-3 xl:grid-cols-6">
        <Stat label={t("evalCenter.group")} value={group.experiment} />
        <Stat label={t("evalCenter.dataset")} value={group.dataset ?? missing} />
        <Stat label={t("evalCenter.methodology")} value={group.methodology ?? missing} hint={group.split ? `${t("evalCenter.split")}: ${group.split}` : undefined} />
        <Stat label={t("evalCenter.evaluatedAt")} value={format.dateTime(group.evaluated_at)} />
        <Stat label={t("evalCenter.checkpoints")} value={group.checkpoints} />
        <Stat label={t("evalCenter.metrics")} value={group.metric_count} />
      </dl>
      <Panel title={t("evalCenter.checkpoint")} actions={<span className="text-xs text-muted-foreground">{t("evalCenter.selected", { count: selected.length })}</span>}>
        <ul className="grid gap-3">
          {group.rows.map((row) => {
            const id = rowId(row);
            return (
              <li key={id} className="grid gap-2 rounded-lg border p-3 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
                <label className="flex min-w-0 items-start gap-2">
                  <Checkbox checked={selected.includes(id)} label={checkpointLabel(row.checkpoint)} onChange={() => onToggle(id)} />
                  <span className="min-w-0">
                    <span className="block truncate text-[13px] font-medium" title={row.checkpoint ?? undefined}>{checkpointLabel(row.checkpoint)}</span>
                    <span className="block text-xs text-muted-foreground">{row.stage ? t(`modelStage.${row.stage}`, { defaultValue: row.stage }) : "—"} · {format.dateTime(row.evaluated_at)} · {Object.keys(row.results).length} {t("evalCenter.metrics").toLowerCase()}</span>
                  </span>
                </label>
                <GateList gates={row.gates} />
              </li>
            );
          })}
        </ul>
      </Panel>
      {others > 0 ? <p className="text-xs text-muted-foreground">{t("evalCenter.otherGroups", { count: others })}</p> : null}
      <ul className="grid gap-1 text-xs text-muted-foreground sm:grid-cols-2 xl:grid-cols-3">
        {SECTIONS.map((section) => {
          const coverage = sectionCoverage(catalog, group.rows, section);
          return <li key={section}>{t(`evalCenter.sections.${section}`)}: {coverage.measured} / {coverage.total}</li>;
        })}
      </ul>
    </div>
  );
}

export function EvaluationPage() {
  const { t } = useTranslation();
  const [search, setSearch] = useState("");
  const [limit, setLimit] = useState(PAGE);
  const [groupId, setGroupId] = useState("");
  const [view, setView] = useState<View>("overview");
  const [picked, setPicked] = useState<string[] | null>(null);
  const catalog = useQuery({ queryKey: ["evaluation-metrics"], queryFn: api.evaluationMetrics, staleTime: Infinity });
  const groups = useQuery({ queryKey: ["evaluation-groups", search, limit], queryFn: () => api.evaluationGroups({ q: search, limit }), staleTime: 30_000, placeholderData: (previous) => previous });
  const current = groups.data?.items.find((item) => item.id === groupId)?.id ?? groups.data?.items[0]?.id ?? "";
  const detail = useQuery({ queryKey: ["evaluation-group", current], queryFn: () => api.evaluationGroup(current), enabled: current !== "", staleTime: 30_000 });
  const rows = useMemo(() => detail.data?.rows ?? [], [detail.data?.rows]);
  const selected = useMemo(() => picked ?? rows.slice(0, DEFAULT_SELECTION).map(rowId), [picked, rows]);
  const chosenRows = rows.filter((row) => selected.includes(rowId(row)));
  const tabs = [{ value: "overview" as View, label: t("evalCenter.overview") }, ...SECTIONS.filter((item) => item !== "efficiency").map((value) => ({ value: value as View, label: t(`evalCenter.sections.${value}`) })), { value: "compare" as View, label: t("evalCenter.compare") }];
  if (groups.isPending || catalog.isPending) return <LoadingState />;
  if (groups.isLoadingError) return <ErrorState error={groups.error} retry={() => void groups.refetch()} />;
  if (catalog.isError) return <ErrorState error={catalog.error} retry={() => void catalog.refetch()} />;
  return (
    <div className="grid gap-6">
      <PageHeader title={t("evaluation.title")} subtitle={t("evaluation.subtitle")} />
      {groups.data.total === 0 && search === "" ? <EmptyState title={t("evalCenter.empty")} description={t("evalCenter.emptyText")} /> : (
        <>
          <div className="flex flex-wrap items-end gap-3">
            <label className="grid min-w-56 flex-1 gap-1 text-xs font-medium">{t("evalCenter.searchGroups")}
              <input className="field" type="search" value={search} onChange={(event) => { setSearch(event.target.value); setLimit(PAGE); }} />
            </label>
            <label className="grid min-w-64 flex-[2] gap-1 text-xs font-medium">{t("evalCenter.group")}
              <Select value={current} onChange={(event) => { setGroupId(event.target.value); setPicked(null); }}>
                {groups.data.items.map((item) => <option key={item.id} value={item.id}>{item.experiment}{item.dataset ? ` · ${item.dataset}` : ""} · {item.checkpoints}</option>)}
              </Select>
            </label>
            {groups.data.items.length < groups.data.total ? <Button variant="outline" size="sm" onClick={() => setLimit((value) => value + PAGE)}>{t("evalCenter.loadMore")}</Button> : null}
          </div>
          {detail.isPending ? <LoadingState /> : detail.isError ? <ErrorState error={detail.error} retry={() => void detail.refetch()} /> : (
            <>
              <Tabs value={view} onChange={setView} items={tabs} />
              {view === "overview" ? <Overview group={detail.data} catalog={catalog.data} selected={selected} onToggle={(id) => setPicked(toggle(selected, id))} others={Math.max(groups.data.total - 1, 0)} /> : null}
              {view === "compare" ? <ComparePanel group={current} checkpoints={selected.map((id) => rows.find((row) => rowId(row) === id)?.checkpoint).filter((item): item is string => Boolean(item))} catalog={catalog.data} /> : null}
              {view !== "overview" && view !== "compare" ? (chosenRows.length === 0 ? <EmptyState title={t("evalCenter.pick")} /> : <MetricSection section={view} catalog={catalog.data} rows={chosenRows} />) : null}
            </>
          )}
        </>
      )}
    </div>
  );
}
