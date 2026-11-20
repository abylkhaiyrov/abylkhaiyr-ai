import { useCallback } from "react";
import { useTranslation } from "react-i18next";
import { useFormat } from "@/hooks/use-format";
import { METRIC_KINDS } from "@/lib/stage";

export function useMetricFormat() {
  const { t } = useTranslation();
  const format = useFormat();
  const label = useCallback((key: string) => t(`metric.${key}`, { defaultValue: key }), [t]);
  const value = useCallback(
    (key: string, raw: number | string | null | undefined) => {
      if (raw === null || raw === undefined) return "—";
      if (typeof raw === "string") return raw;
      switch (METRIC_KINDS[key] ?? "num") {
        case "int":
          return format.number(raw, 0);
        case "compact":
          return format.compact(raw);
        case "loss":
          return format.number(raw, 4);
        case "ppl":
          return format.number(raw, 2);
        case "pct":
          return format.percent(raw * 100, 1);
        case "sci":
          return new Intl.NumberFormat(undefined, { notation: "scientific", maximumFractionDigits: 2 }).format(raw);
        case "rate":
          return `${format.number(raw, 0)} ${t("units.tokensPerSecond")}`;
        case "text":
          return String(raw);
        default:
          return format.number(raw, 3);
      }
    },
    [format, t],
  );
  return { label, value };
}
