import { CheckCircle2, CircleDashed, XCircle } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { GateStatus, QualityGate } from "@/lib/api";
import { cn } from "@/lib/utils";

const ICONS = { passed: CheckCircle2, failed: XCircle, not_verified: CircleDashed } as const;
const TONES: Record<GateStatus, string> = { passed: "text-success", failed: "text-destructive", not_verified: "text-muted-foreground" };

export function GateList({ gates }: { gates: QualityGate[] }) {
  const { t } = useTranslation();
  return (
    <ul className="grid gap-1.5 text-[13px]">
      {gates.map((gate) => {
        const Icon = ICONS[gate.status];
        return (
          <li key={gate.id} className="flex items-start gap-2">
            <Icon size={15} className={cn("mt-0.5 shrink-0", TONES[gate.status])} aria-hidden />
            <span><span className={TONES[gate.status]}>{t(`evalCenter.gateStatus.${gate.status}`)}</span> · {t(`evalCenter.gate.${gate.id}`, { value: gate.threshold })}</span>
          </li>
        );
      })}
    </ul>
  );
}
