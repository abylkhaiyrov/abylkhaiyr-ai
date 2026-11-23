import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";
import { PageHeader, Panel } from "@/components/data-state";
import { useCurrentUser } from "@/hooks/use-auth";
import { Select } from "@/components/ui/select";
import { useUiStore } from "@/store/ui-store";

export function SettingsPage() {
  const { t } = useTranslation();
  const user = useCurrentUser();
  const { locale, setLocale, theme, setTheme } = useUiStore();
  return (
    <div className="grid max-w-2xl gap-8">
      <PageHeader title={t("settings.title")} subtitle={t("settings.subtitle")} />
      <Panel title={t("settings.appearance")}>
        <div className="grid gap-4 sm:grid-cols-2">
          <label className="grid gap-1.5 text-[13px] font-medium">{t("settings.language")}
            <Select className="h-9" value={locale} onChange={(event) => setLocale(event.target.value as "ru" | "en")}>
              <option value="ru">Русский</option>
              <option value="en">English</option>
            </Select>
          </label>
          <label className="grid gap-1.5 text-[13px] font-medium">{t("settings.theme")}
            <Select className="h-9" value={theme} onChange={(event) => setTheme(event.target.value as "light" | "dark")}>
              <option value="light">{t("settings.light")}</option>
              <option value="dark">{t("settings.dark")}</option>
            </Select>
          </label>
        </div>
      </Panel>
      <Panel title={t("settings.account")}>
        <p className="text-[13px] text-muted-foreground">{`${user.first_name} ${user.last_name}`.trim()} · @{user.username} · {t(`role.account.${user.role}`)}</p>
        <Link className="mt-2 inline-block text-[13px] font-medium text-accent hover:underline" to="/profile">{t("settings.manageAccount")}</Link>
      </Panel>
    </div>
  );
}
