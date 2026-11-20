import { useState, type FormEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { PageHeader, Panel } from "@/components/data-state";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { useToastStore } from "@/store/toast-store";
import { useUiStore } from "@/store/ui-store";

export function SettingsPage() {
  const { t } = useTranslation();
  const client = useQueryClient();
  const push = useToastStore((state) => state.push);
  const { locale, setLocale, theme, setTheme } = useUiStore();
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const change = useMutation({
    mutationFn: () => api.changePassword(currentPassword, newPassword),
    onSuccess: () => { push("success", t("auth.passwordUpdated")); client.clear(); },
  });
  const submit = (event: FormEvent<HTMLFormElement>) => { event.preventDefault(); change.mutate(); };
  return (
    <div className="grid max-w-2xl gap-4">
      <PageHeader title={t("settings.title")} subtitle={t("settings.subtitle")} />
      <Panel title={t("settings.appearance")}>
        <div className="grid gap-4 sm:grid-cols-2">
          <label className="grid gap-1.5 text-sm font-medium">{t("settings.language")}
            <select className="field" value={locale} onChange={(event) => setLocale(event.target.value as "ru" | "en")}>
              <option value="ru">Русский</option>
              <option value="en">English</option>
            </select>
          </label>
          <label className="grid gap-1.5 text-sm font-medium">{t("settings.theme")}
            <select className="field" value={theme} onChange={(event) => setTheme(event.target.value as "light" | "dark")}>
              <option value="light">{t("settings.light")}</option>
              <option value="dark">{t("settings.dark")}</option>
            </select>
          </label>
        </div>
      </Panel>
      <Panel title={t("settings.security")}>
        <form className="grid gap-3" onSubmit={submit}>
          <label className="grid gap-1.5 text-sm font-medium">{t("auth.currentPassword")}
            <input type="password" autoComplete="current-password" className="field" value={currentPassword} onChange={(event) => setCurrentPassword(event.target.value)} />
          </label>
          <label className="grid gap-1.5 text-sm font-medium">{t("auth.newPassword")}
            <input type="password" autoComplete="new-password" minLength={12} className="field" value={newPassword} onChange={(event) => setNewPassword(event.target.value)} />
            <span className="text-xs font-normal text-muted-foreground">{t("auth.passwordHint")}</span>
          </label>
          {change.error ? <p role="alert" className="text-sm text-destructive">{change.error.message}</p> : null}
          <Button className="w-fit" type="submit" disabled={change.isPending}>{t("auth.changePassword")}</Button>
        </form>
      </Panel>
    </div>
  );
}
