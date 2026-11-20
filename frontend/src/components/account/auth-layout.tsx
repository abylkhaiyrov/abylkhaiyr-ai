import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { Toaster } from "@/components/toaster";
import { useUiStore } from "@/store/ui-store";

export function AuthLayout({ title, subtitle, children }: { title: string; subtitle: string; children: ReactNode }) {
  const { t } = useTranslation();
  const { locale, setLocale } = useUiStore();
  return (
    <main className="grid min-h-dvh place-items-center p-4">
      <div className="absolute right-4 top-4">
        <select className="h-8 rounded-md bg-transparent px-2 text-xs text-muted-foreground hover:bg-muted" value={locale} onChange={(event) => setLocale(event.target.value as "ru" | "en")} aria-label={t("settings.language")}>
          <option value="ru">RU</option>
          <option value="en">EN</option>
        </select>
      </div>
      <div className="w-full max-w-sm">
        <div className="mb-8">
          <div className="grid h-9 w-9 place-items-center rounded-lg bg-accent text-sm font-bold text-accent-foreground">A</div>
          <h1 className="mt-5 text-2xl font-semibold tracking-tight">{title}</h1>
          <p className="mt-1 text-[13px] text-muted-foreground">{subtitle}</p>
        </div>
        {children}
      </div>
      <Toaster />
    </main>
  );
}
