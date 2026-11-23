import { useTranslation } from "react-i18next";
import { Select } from "@/components/ui/select";
import type { InferenceProfile, ProfileId } from "@/lib/api";
import { PROFILE_ORDER } from "@/lib/profiles";

type ProfilePickerProps = {
  profiles: InferenceProfile[];
  value: ProfileId | null;
  locked: boolean;
  onChange: (id: ProfileId) => void;
};

export function ProfilePicker({ profiles, value, locked, onChange }: ProfilePickerProps) {
  const { t } = useTranslation();
  const ordered = PROFILE_ORDER.flatMap((id) => profiles.filter((item) => item.id === id));
  const current = ordered.find((item) => item.id === value);
  return (
    <span className="inline-flex" title={current ? t("profile.summary", { tokens: current.max_output_tokens, seconds: current.generation_timeout_seconds }) : undefined}>
      <Select compact aria-label={t("profile.label")} disabled={locked || ordered.length === 0} value={value ?? ""} onChange={(event) => onChange(event.target.value as ProfileId)}>
        {value === null ? <option value="" disabled>{t("profile.choose")}</option> : null}
        {ordered.map((item) => (
          <option key={item.id} value={item.id} disabled={!item.available || !item.allowed}>
            {t(`profile.names.${item.id}`)}{!item.available ? ` · ${t(`profile.reasons.${item.unavailable_reason}`, { defaultValue: t("chat.unavailable") })}` : !item.allowed ? ` · ${t("profile.disabled")}` : ""}
          </option>
        ))}
      </Select>
    </span>
  );
}
