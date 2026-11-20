import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { isFiniteNumber } from "@/lib/utils";

const DASH = "—";
const BYTE_UNITS = ["b", "kb", "mb", "gb", "tb"] as const;

export function useFormat() {
  const { t, i18n } = useTranslation();
  const language = i18n.language;
  return useMemo(() => {
    const number = (value: number | null | undefined, digits = 2) =>
      isFiniteNumber(value)
        ? new Intl.NumberFormat(language, { maximumFractionDigits: digits }).format(value)
        : DASH;
    const compact = (value: number | null | undefined) =>
      isFiniteNumber(value)
        ? new Intl.NumberFormat(language, { notation: "compact", maximumFractionDigits: 2 }).format(value)
        : DASH;
    const percent = (value: number | null | undefined, digits = 1) =>
      isFiniteNumber(value) ? `${number(value, digits)}%` : DASH;
    const bytes = (value: number | null | undefined) => {
      if (!isFiniteNumber(value)) return DASH;
      const exponent = value < 1 ? 0 : Math.min(Math.floor(Math.log(value) / Math.log(1024)), BYTE_UNITS.length - 1);
      const scaled = value / 1024 ** exponent;
      return `${number(scaled, exponent >= 3 ? 2 : scaled >= 100 ? 0 : 1)} ${t(`units.${BYTE_UNITS[exponent]}`)}`;
    };
    const duration = (seconds: number | null | undefined) => {
      if (!isFiniteNumber(seconds)) return DASH;
      const total = Math.max(0, Math.round(seconds));
      const days = Math.floor(total / 86400);
      const hours = Math.floor((total % 86400) / 3600);
      const minutes = Math.floor((total % 3600) / 60);
      const rest = total % 60;
      const parts = days > 0
        ? [`${days} ${t("units.d")}`, `${hours} ${t("units.h")}`]
        : hours > 0
          ? [`${hours} ${t("units.h")}`, `${minutes} ${t("units.m")}`]
          : minutes > 0
            ? [`${minutes} ${t("units.m")}`, `${rest} ${t("units.s")}`]
            : [`${rest} ${t("units.s")}`];
      return parts.join(" ");
    };
    const dateTime = (value: string | number | null | undefined) =>
      value === null || value === undefined
        ? DASH
        : new Intl.DateTimeFormat(language, { dateStyle: "medium", timeStyle: "short" }).format(new Date(value));
    const time = (value: string | number | null | undefined) =>
      value === null || value === undefined
        ? DASH
        : new Intl.DateTimeFormat(language, { timeStyle: "medium" }).format(new Date(value));
    const clock = (value: string | number | null | undefined) =>
      value === null || value === undefined
        ? DASH
        : new Intl.DateTimeFormat(language, { hour: "2-digit", minute: "2-digit" }).format(new Date(value));
    const relative = (value: string | number | null | undefined) => {
      if (value === null || value === undefined) return DASH;
      const stamp = typeof value === "string" && !/[zZ]|[+-]\d\d:?\d\d$/.test(value) ? `${value}Z` : value;
      const seconds = Math.round((new Date(stamp).getTime() - Date.now()) / 1000);
      const formatter = new Intl.RelativeTimeFormat(language, { numeric: "auto" });
      const steps: [Intl.RelativeTimeFormatUnit, number][] = [["day", 86400], ["hour", 3600], ["minute", 60]];
      for (const [unit, size] of steps) if (Math.abs(seconds) >= size) return formatter.format(Math.round(seconds / size), unit);
      return formatter.format(seconds, "second");
    };
    const ratio = (done: number | null | undefined, total: number | null | undefined) =>
      isFiniteNumber(done) && isFiniteNumber(total) ? `${number(done, 0)} / ${number(total, 0)}` : DASH;
    return { number, compact, percent, bytes, duration, dateTime, time, clock, relative, ratio };
  }, [language, t]);
}
