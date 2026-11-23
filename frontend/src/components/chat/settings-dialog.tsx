import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { DEFAULT_SETTINGS } from "@/lib/chat-settings";
import { Select } from "@/components/ui/select";
import type { GenerationSettings } from "@/lib/api";

const MODES = ["standard", "reasoning", "precise", "creative"] as const;

type Props = {
  open: boolean;
  settings: Required<GenerationSettings>;
  systemPrompt: string;
  onClose: () => void;
  onSave: (settings: Required<GenerationSettings>, systemPrompt: string) => void;
};

const FIELDS = [
  ["temperature", "chat.temperature", 0, 2, 0.05],
  ["top_p", "chat.topP", 0.05, 1, 0.05],
  ["top_k", "chat.topK", 0, 200, 1],
  ["max_tokens", "chat.maxTokens", 16, 512, 16],
  ["repetition_penalty", "chat.repetitionPenalty", 1, 2, 0.05],
] as const;

export function SettingsDialog({ open, settings, systemPrompt, onClose, onSave }: Props) {
  const { t } = useTranslation();
  const [draft, setDraft] = useState(settings);
  const [prompt, setPrompt] = useState(systemPrompt);
  return (
    <Dialog
      open={open}
      title={t("chat.settings")}
      onClose={onClose}
      footer={
        <>
          <Button variant="ghost" onClick={() => { setDraft(DEFAULT_SETTINGS); setPrompt(""); }}>{t("common.reset")}</Button>
          <Button variant="outline" onClick={onClose}>{t("common.cancel")}</Button>
          <Button onClick={() => { onSave(draft, prompt); onClose(); }}>{t("common.save")}</Button>
        </>
      }
    >
      <div className="grid gap-4">
        <label className="grid gap-1.5 text-sm font-medium">{t("chat.mode")}
          <Select value={draft.mode} onChange={(event) => setDraft({ ...draft, mode: event.target.value as typeof draft.mode })}>
            {MODES.map((mode) => <option key={mode} value={mode}>{t(`chat.modes.${mode}`)}</option>)}
          </Select>
          <span className="text-xs font-normal text-muted-foreground">{t(`chat.modeHelp.${draft.mode}`)}</span>
        </label>
        {FIELDS.map(([key, label, min, max, step]) => (
          <label key={key} className="grid gap-1 text-sm font-medium">
            <span className="flex justify-between">{t(label)}<span className="font-normal tabular-nums text-muted-foreground">{draft[key]}</span></span>
            <input type="range" min={min} max={max} step={step} value={draft[key]} onChange={(event) => setDraft({ ...draft, [key]: Number(event.target.value) })} className="accent-[hsl(var(--primary))]" />
          </label>
        ))}
        <label className="grid gap-1.5 text-sm font-medium">{t("chat.systemPrompt")}
          <textarea className="field min-h-20 resize-y py-2 font-normal" placeholder={t("chat.systemPromptPlaceholder")} value={prompt} onChange={(event) => setPrompt(event.target.value)} />
        </label>
      </div>
    </Dialog>
  );
}
