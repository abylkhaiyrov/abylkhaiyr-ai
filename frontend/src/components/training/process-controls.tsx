import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Pause, Play, Square } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Panel } from "@/components/data-state";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/dialog";
import { StatusBadge } from "@/components/status-badge";
import { useFormat } from "@/hooks/use-format";
import { useRuntime } from "@/hooks/use-runtime";
import { useApiError } from "@/hooks/use-api-error";
import { api } from "@/lib/api";
import { ApiError } from "@/lib/http";
import { useToastStore } from "@/store/toast-store";

type StopTarget = { kind: "pipeline" } | { kind: "process"; id: string } | null;

function reasonKey(error: unknown) {
  return error instanceof ApiError && error.detail && /^[a-z_]+$/.test(error.detail) ? `training.errors.${error.detail}` : null;
}

export function ProcessControls() {
  const { t } = useTranslation();
  const format = useFormat();
  const client = useQueryClient();
  const push = useToastStore((state) => state.push);
  const describe = useApiError();
  const runtime = useRuntime().data;
  const busy = runtime?.compute.busy ?? false;
  const processes = useQuery({ queryKey: ["processes"], queryFn: api.processes, staleTime: 30_000 });
  const configs = useQuery({ queryKey: ["training-configs"], queryFn: api.trainingConfigs, staleTime: 300_000 });
  const [config, setConfig] = useState("");
  const [stop, setStop] = useState<StopTarget>(null);
  const holder = runtime?.compute.holder;
  const settle = () => {
    void client.invalidateQueries({ queryKey: ["processes"] });
    void client.invalidateQueries({ queryKey: ["runtime"] });
  };
  const fail = (error: Error) => {
    const key = reasonKey(error);
    push("error", key ? t(key) : describe(error));
  };
  const start = useMutation({ mutationFn: () => api.startTraining(config), onSuccess: () => { push("success", t("training.started")); settle(); }, onError: fail });
  const pause = useMutation({ mutationFn: api.pauseTraining, onSuccess: settle, onError: fail });
  const stopProcess = useMutation({ mutationFn: async (target: NonNullable<StopTarget>) => {
      if (target.kind === "pipeline") await api.stopPipeline();
      else await api.stopTraining(target.id);
    }, onSuccess: () => { setStop(null); push("info", t("training.stopRequested")); settle(); }, onError: fail });
  const managed = (processes.data ?? []).filter((item) => item.managed !== false && ["running", "pausing", "paused"].includes(item.status));
  const activePipeline = runtime?.pipeline?.running ? runtime.pipeline : null;
  const blockReason = busy && holder ? t("training.blocked", { kind: t(`holder.${holder.kind}`), pid: holder.pid ?? "—", stage: holder.stage ? t(`stage.${holder.stage}`, { defaultValue: holder.stage }) : "" }) : null;
  const launchable = (configs.data ?? []).filter((item) => item.path.endsWith(".yaml"));
  return (
    <Panel title={t("training.control")}>
      <div className="grid gap-4">
        <div className="grid gap-2 sm:grid-cols-[minmax(0,1fr)_auto]">
          <select aria-label={t("training.config")} className="field" value={config} onChange={(event) => setConfig(event.target.value)}>
            <option value="">{t("training.chooseConfig")}</option>
            {launchable.map((item) => <option key={item.path} value={item.path}>{item.name} · {t(`modelStage.${item.stage}`)}</option>)}
          </select>
          <Button disabled={busy || config === "" || start.isPending} onClick={() => start.mutate()}><Play size={14} />{t("training.launch")}</Button>
        </div>
        {blockReason ? <p role="status" className="text-xs text-warning">{blockReason}</p> : null}
        {activePipeline ? (
          <div className="flex flex-wrap items-center justify-between gap-2 border-t hairline pt-3">
            <div className="min-w-0">
              <p className="font-medium">{t("training.pipelineRunning", { name: activePipeline.model_name })}</p>
              <p className="text-xs text-muted-foreground">PID {activePipeline.pid} · {t("training.cpu")} {format.number(activePipeline.cpu_percent, 0)}% · {format.bytes(activePipeline.memory_bytes)}</p>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" disabled title={t("training.pauseUnsupported")}><Pause size={13} />{t("training.pause")}</Button>
              <Button variant="danger" size="sm" onClick={() => setStop({ kind: "pipeline" })}><Square size={13} />{t("training.stop")}</Button>
            </div>
          </div>
        ) : null}
        {managed.map((process) => (
          <div key={process.id} className="flex flex-wrap items-center justify-between gap-2 border-t hairline pt-3">
            <div className="min-w-0">
              <p className="flex items-center gap-2 font-medium">{t(`modelStage.${process.kind}`, { defaultValue: process.kind })}<StatusBadge value={process.status} /></p>
              <p className="text-xs text-muted-foreground">PID {process.pid ?? "—"} · {process.config_path}</p>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" disabled={process.status !== "running" || pause.isPending} onClick={() => pause.mutate(process.id)}><Pause size={13} />{t("training.pause")}</Button>
              <Button variant="danger" size="sm" onClick={() => setStop({ kind: "process", id: process.id })}><Square size={13} />{t("training.stop")}</Button>
            </div>
          </div>
        ))}
      </div>
      <ConfirmDialog
        open={stop !== null}
        danger
        busy={stopProcess.isPending}
        title={t("training.stopTitle")}
        description={stop?.kind === "pipeline" ? t("training.stopPipelineText") : t("training.stopProcessText")}
        confirmLabel={t("training.stop")}
        onClose={() => setStop(null)}
        onConfirm={() => stop && stopProcess.mutate(stop)}
      />
    </Panel>
  );
}
