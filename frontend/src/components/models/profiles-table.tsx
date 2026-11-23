import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { Checkbox } from "@/components/training/checkbox";
import { useApiError } from "@/hooks/use-api-error";
import { api, type InferenceProfile } from "@/lib/api";
import { PROFILE_ORDER } from "@/lib/profiles";
import { useToastStore } from "@/store/toast-store";

export function ProfilesTable({ profiles }: { profiles: InferenceProfile[] }) {
  const { t } = useTranslation();
  const client = useQueryClient();
  const push = useToastStore((state) => state.push);
  const describe = useApiError();
  const save = useMutation({
    mutationFn: api.setAllowedProfiles,
    onSuccess: (next) => client.setQueryData(["inference-profiles"], next),
    onError: (error: Error) => push("error", describe(error)),
  });
  const ordered = PROFILE_ORDER.flatMap((id) => profiles.filter((item) => item.id === id));
  const toggle = (profile: InferenceProfile) => save.mutate(ordered.filter((item) => (item.id === profile.id ? !item.allowed : item.allowed)).map((item) => item.id));
  return (
    <div className="grid gap-3">
      <p className="max-w-2xl text-[13px] text-muted-foreground">{t("profile.note")}</p>
      <div className="overflow-x-auto border-y hairline">
        <table className="data-table">
          <thead><tr><th>{t("profile.label")}</th><th>{t("profile.budget")}</th><th>{t("profile.context")}</th><th>{t("profile.timeout")}</th><th>{t("profile.tools")}</th><th>{t("profile.verification")}</th><th>{t("profile.reasoning")}</th><th>{t("models.visibleToUsers")}</th></tr></thead>
          <tbody>
            {ordered.map((profile) => (
              <tr key={profile.id}>
                <td><p className="font-medium">{t(`profile.names.${profile.id}`)}</p><p className="max-w-64 text-xs text-muted-foreground">{t(`profile.help.${profile.id}`)}</p></td>
                <td className="tabular-nums">{profile.max_output_tokens}</td>
                <td className="tabular-nums">{profile.context_budget}</td>
                <td className="tabular-nums">{t("profile.seconds", { value: profile.generation_timeout_seconds })}</td>
                <td className="tabular-nums">{profile.tool_budget_seconds > 0 ? t("profile.seconds", { value: profile.tool_budget_seconds }) : "—"}</td>
                <td className="tabular-nums">{profile.reasoning.verification_attempts}</td>
                <td>{profile.reasoning.enabled ? t("profile.on") : t("profile.off")}{profile.reasoning.enabled ? <span className="ml-1 text-xs text-muted-foreground">×{profile.reasoning.max_candidates}</span> : null}</td>
                <td>
                  <label className="inline-flex items-center gap-2 text-xs">
                    <Checkbox checked={profile.allowed} disabled={save.isPending} onChange={() => toggle(profile)} label={t("models.visibleToUsers")} />
                    {!profile.available ? <span className="text-warning">{t(`profile.reasons.${profile.unavailable_reason}`, { defaultValue: t("chat.unavailable") })}</span> : null}
                  </label>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
