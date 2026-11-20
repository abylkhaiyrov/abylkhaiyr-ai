import { useTranslation } from "react-i18next";
import { EmptyState } from "@/components/data-state";
import { useFormat } from "@/hooks/use-format";
import { useMetricFormat } from "@/hooks/use-metric-format";
import type { Metric, RunDetail } from "@/lib/api";

const COLUMNS = ["test_perplexity", "next_token_accuracy", "instruction_following_accuracy"];

export function CheckpointsTable({ run }: { run: RunDetail }) {
  const { t } = useTranslation();
  const format = useFormat();
  const metric = useMetricFormat();
  if (run.checkpoints.length === 0) return <EmptyState title={t("training.noCheckpoints")} />;
  const evaluation = (label: string): Metric | undefined => run.evaluations.find((item) => item.checkpoint === label)?.results;
  return (
    <div className="overflow-x-auto rounded-xl border">
      <table className="data-table">
        <thead>
          <tr>
            <th>{t("training.checkpoint")}</th>
            <th>{t("metric.step")}</th>
            <th>{t("metric.tokens_seen")}</th>
            <th>{t("training.created")}</th>
            {COLUMNS.map((key) => <th key={key}>{metric.label(key)}</th>)}
          </tr>
        </thead>
        <tbody>
          {[...run.checkpoints].reverse().map((checkpoint) => {
            const results = evaluation(checkpoint.label);
            return (
              <tr key={checkpoint.id}>
                <td className="font-medium">{checkpoint.label}</td>
                <td className="tabular-nums">{format.number(checkpoint.step, 0)}</td>
                <td className="tabular-nums">{format.compact(checkpoint.tokens_seen)}</td>
                <td>{format.dateTime(checkpoint.created_at)}</td>
                {COLUMNS.map((key) => <td key={key} className="tabular-nums">{results ? metric.value(key, results[key]) : "—"}</td>)}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
