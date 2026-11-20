import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { EmptyState, ErrorState, LoadingState, PageHeader, Panel } from "@/components/data-state";
import { useMetricFormat } from "@/hooks/use-metric-format";
import { api, type Evaluation } from "@/lib/api";

const CATEGORIES = {
  language: ["test_perplexity", "test_loss", "next_token_accuracy", "perplexity_ru", "perplexity_kk", "perplexity_en"],
  instructions: ["instruction_following_accuracy", "multiturn_dialogue_accuracy", "context_retrieval_accuracy"],
  reasoning: ["arithmetic_exact_match", "math_bench_model_accuracy", "math_bench_tool_accuracy", "logic_model_accuracy", "logic_tool_accuracy"],
  code: ["code_pass_at_1", "code_pass_at_4", "code_tool_pass_rate"],
  facts: ["factual_accuracy", "factual_wrong_rate"],
  response: ["non_empty_response_rate", "coherence_distinct_bigram", "repeated_trigram_ratio_greedy", "generation_tokens_per_second"],
} as const;

function Filter({ label, value, options, onChange }: {
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="grid min-w-0 gap-1 text-xs font-medium">
      {label}
      <select aria-label={label} className="field" value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
      </select>
    </label>
  );
}

const optionsFor = (values: (string | null)[]) =>
  [...new Set(values.filter((value): value is string => Boolean(value)))].map((value) => ({ value, label: value }));
const checkpointLabel = (value: string) => value.split("/").slice(-2).join("/");
const reportKey = (report: Evaluation) => JSON.stringify([report.path, report.dataset, report.stage, report.checkpoint]);

function EvaluationComparison({ reports }: { reports: Evaluation[] }) {
  const { t } = useTranslation();
  const metric = useMetricFormat();
  const [selection, setSelection] = useState("");
  const [model, setModel] = useState("");
  const [checkpoint, setCheckpoint] = useState("");
  const [category, setCategory] = useState<keyof typeof CATEGORIES>("language");
  const [language, setLanguage] = useState("");
  const [chartMetric, setChartMetric] = useState("");
  const cohort = (report: Evaluation) => JSON.stringify([report.experiment, report.dataset, report.path]);
  const cohorts = [...new Map(reports.map((report) => [cohort(report), {
    value: cohort(report), label: [report.experiment, report.dataset].filter(Boolean).join(" · "),
  }])).values()].sort((left, right) => Number(right.label.startsWith("pipeline_")) - Number(left.label.startsWith("pipeline_")));
  const current = cohorts.find((item) => item.value === selection)?.value ?? cohorts[0].value;
  const candidates = reports.filter((report) => cohort(report) === current);
  const models = candidates.filter((report) => !model || report.model === model);
  const rows = models.filter((report) => !checkpoint || report.checkpoint === checkpoint);
  const metrics = CATEGORIES[category].filter((key) => {
    if (category === "language" && language && key !== `perplexity_${language}`) return false;
    return rows.some((row) => row.results[key] !== undefined);
  });
  const activeMetric = metrics.find((key) => key === chartMetric) ?? metrics[0];
  const max = activeMetric ? Math.max(0, ...rows.map((row) => row.results[activeMetric] ?? 0)) : 0;
  const all = { value: "", label: t("evaluation.all") };
  return (
    <>
      <div className="grid gap-3 rounded-xl border p-4 sm:grid-cols-2 xl:grid-cols-5">
        <Filter label={t("evaluation.experiment")} value={current} options={cohorts} onChange={(value) => {
          setSelection(value);
          setModel("");
          setCheckpoint("");
        }} />
        <Filter label={t("models.model")} value={model} options={[all, ...optionsFor(candidates.map((row) => row.model))]} onChange={(value) => {
          setModel(value);
          setCheckpoint("");
        }} />
        <Filter label={t("evaluation.checkpoint")} value={checkpoint} options={[all, ...optionsFor(models.map((row) => row.checkpoint)).map((item) => ({ ...item, label: checkpointLabel(item.value) }))]} onChange={setCheckpoint} />
        <Filter label={t("evaluation.category")} value={category} options={Object.keys(CATEGORIES).map((value) => ({ value, label: t(`evaluationCategory.${value}`) }))} onChange={(value) => {
          if (Object.hasOwn(CATEGORIES, value)) setCategory(value as keyof typeof CATEGORIES);
        }} />
        <Filter label={t("evaluation.language")} value={language} options={[all, ...["ru", "kk", "en"].map((value) => ({ value, label: t(`evaluationLanguage.${value}`) }))]} onChange={(value) => {
          setLanguage(value);
          setCategory("language");
        }} />
      </div>
      {metrics.length === 0 ? <EmptyState title={t("evaluation.noMetrics")} /> : (
        <>
          <Panel title={t("evaluation.comparison")} className="min-w-0">
            <div className="overflow-x-auto rounded-xl border">
              <table className="data-table">
                <thead><tr>
                  <th scope="col">{t("evaluation.checkpoint")}</th>
                  <th scope="col">{t("models.stage")}</th>
                  {metrics.map((key) => <th scope="col" key={key}>{metric.label(key)}</th>)}
                </tr></thead>
                <tbody>
                  {rows.map((row) => (
                    <tr key={reportKey(row)}>
                      <th scope="row" className="text-left">{row.checkpoint ? checkpointLabel(row.checkpoint) : row.experiment}</th>
                      <td>{row.stage ? t(`modelStage.${row.stage}`, { defaultValue: row.stage }) : "—"}</td>
                      {metrics.map((key) => <td className="tabular-nums" key={key}>{metric.value(key, row.results[key])}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
          <Panel title={t("evaluation.chart")} className="grid gap-3 rounded-xl border p-4">
            <Filter label={t("evaluation.metric")} value={activeMetric} options={metrics.map((value) => ({ value, label: metric.label(value) }))} onChange={setChartMetric} />
            <ul className="grid gap-2">
              {rows.filter((row) => row.results[activeMetric] !== undefined).map((row) => {
                const value = row.results[activeMetric];
                return (
                  <li key={reportKey(row)} className="grid gap-1">
                    <div className="flex flex-wrap justify-between gap-x-3 text-xs">
                      <span>{row.checkpoint ? checkpointLabel(row.checkpoint) : row.experiment}</span>
                      <span className="tabular-nums">{metric.value(activeMetric, value)}</span>
                    </div>
                    <div aria-hidden className="h-2 rounded-full bg-muted">
                      <div className="h-full rounded-full bg-accent" style={{ width: `${max > 0 ? value / max * 100 : 0}%` }} />
                    </div>
                  </li>
                );
              })}
            </ul>
          </Panel>
        </>
      )}
    </>
  );
}

export function EvaluationPage() {
  const { t } = useTranslation();
  const query = useQuery({ queryKey: ["evaluation"], queryFn: api.evaluations });
  return (
    <div className="grid gap-5">
      <PageHeader title={t("evaluation.title")} subtitle={t("evaluation.subtitle")} />
      {query.isPending ? <LoadingState /> : query.isError ? (
        <ErrorState error={query.error} retry={() => void query.refetch()} />
      ) : query.data.length === 0 ? (
        <EmptyState title={t("evaluation.empty")} />
      ) : <EvaluationComparison reports={query.data} />}
    </div>
  );
}
