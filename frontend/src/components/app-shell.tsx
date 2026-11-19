import {
  Activity,
  BarChart3,
  Bot,
  Boxes,
  ClipboardList,
  Database,
  FileText,
  FlaskConical,
  LayoutDashboard,
  LogOut,
  Menu,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  Settings,
  Sun,
  TerminalSquare,
  X,
} from "lucide-react";
import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { useUiStore } from "@/store/ui-store";

const primaryNavigation = [
  ["/", "overview", LayoutDashboard],
  ["/chat", "chat", Bot],
  ["/training", "training", Activity],
  ["/monitoring", "monitoring", BarChart3],
] as const;

const managementNavigation = [
  ["/models", "models", Boxes],
  ["/datasets", "datasets", Database],
  ["/experiments", "experiments", FlaskConical],
  ["/evaluation", "evaluation", ClipboardList],
  ["/processes", "processes", TerminalSquare],
  ["/logs", "logs", FileText],
] as const;

function Navigation({ close, collapsed = false }: { close: () => void; collapsed?: boolean }) {
  const { t } = useTranslation();
  return (
    <nav aria-label={t("app.name")} className="flex min-h-0 flex-1 flex-col gap-6 overflow-y-auto p-3">
      <div className="grid gap-1">
        {primaryNavigation.map(([to, label, Icon]) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            onClick={close}
            aria-label={t(`nav.${label}`)}
            title={collapsed ? t(`nav.${label}`) : undefined}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors ${isActive ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted hover:text-foreground"}`
            }
          >
            <Icon size={17} />
            {!collapsed && t(`nav.${label}`)}
          </NavLink>
        ))}
      </div>
      <div>
        <p className={`px-3 pb-2 text-[11px] font-semibold uppercase tracking-widest text-muted-foreground ${collapsed ? "sr-only" : ""}`}>
          {t("common.details")}
        </p>
        <div className="grid gap-1">
          {managementNavigation.map(([to, label, Icon]) => (
            <NavLink
              key={to}
              to={to}
              onClick={close}
              aria-label={t(`nav.${label}`)}
              title={collapsed ? t(`nav.${label}`) : undefined}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors ${isActive ? "bg-muted font-medium text-foreground" : "text-muted-foreground hover:bg-muted hover:text-foreground"}`
              }
            >
              <Icon size={16} />
              {!collapsed && t(`nav.${label}`)}
            </NavLink>
          ))}
        </div>
      </div>
      <NavLink
        to="/settings"
        onClick={close}
        aria-label={t("nav.settings")}
        title={collapsed ? t("nav.settings") : undefined}
        className={({ isActive }) =>
          `mt-auto flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors ${isActive ? "bg-muted font-medium text-foreground" : "text-muted-foreground hover:bg-muted hover:text-foreground"}`
        }
      >
        <Settings size={16} />
        {!collapsed && t("nav.settings")}
      </NavLink>
    </nav>
  );
}

export function AppShell() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const client = useQueryClient();
  const [mobileOpen, setMobileOpen] = useState(false);
  const { theme, locale, backendOnline, lastUpdated, setTheme, setLocale, sidebarCollapsed, toggleSidebar } = useUiStore();
  const logout = useMutation({
    mutationFn: api.logout,
    onSuccess: () => {
      client.clear();
      navigate("/login");
    },
  });

  return (
    <div className={`min-h-screen bg-background lg:grid ${sidebarCollapsed ? "lg:grid-cols-[4rem_minmax(0,1fr)]" : "lg:grid-cols-[15rem_minmax(0,1fr)]"}`}>
      <aside className="hidden border-r bg-card lg:flex lg:min-h-screen lg:flex-col">
        <div className={`flex h-16 items-center gap-3 border-b ${sidebarCollapsed ? "justify-center px-2" : "px-5"}`}>
          <div className="grid h-8 w-8 place-items-center rounded-lg bg-primary text-sm font-bold text-primary-foreground">A</div>
          {!sidebarCollapsed && <span className="text-sm font-semibold tracking-tight">{t("app.name")}</span>}
        </div>
        <Navigation close={() => undefined} collapsed={sidebarCollapsed} />
      </aside>
      {mobileOpen ? (
        <div className="fixed inset-0 z-50 bg-black/45 lg:hidden" onClick={() => setMobileOpen(false)}>
          <aside className="flex h-full w-72 flex-col bg-card shadow-xl" onClick={(event) => event.stopPropagation()}>
            <div className="flex h-16 items-center justify-between border-b px-5">
              <span className="text-sm font-semibold">{t("app.name")}</span>
              <Button className="h-9 w-9 px-0" variant="ghost" aria-label={t("common.close")} onClick={() => setMobileOpen(false)}><X size={18} /></Button>
            </div>
            <Navigation close={() => setMobileOpen(false)} />
          </aside>
        </div>
      ) : null}
      <main className="min-w-0">
        <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b bg-background/90 px-4 backdrop-blur lg:px-6">
          <div className="flex items-center gap-2">
            <Button className="h-9 w-9 px-0 lg:hidden" variant="ghost" aria-label={t("common.details")} onClick={() => setMobileOpen(true)}><Menu size={18} /></Button>
            <Button className="hidden h-9 w-9 px-0 lg:inline-flex" variant="ghost" aria-label={t(sidebarCollapsed ? "app.expandSidebar" : "app.collapseSidebar")} onClick={toggleSidebar}>{sidebarCollapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}</Button>
            <div className="hidden items-center gap-2 text-xs text-muted-foreground sm:flex"><span className={`h-2 w-2 rounded-full ${backendOnline ? "bg-emerald-500" : "bg-destructive"}`} /><span>{backendOnline ? t("app.connected") : t("app.offline")}</span>{lastUpdated ? <span>{t("app.updated", { time: new Date(lastUpdated).toLocaleTimeString() })}</span> : null}</div>
          </div>
          <div className="flex items-center gap-1">
            <label className="sr-only" htmlFor="language">{t("settings.language")}</label>
            <select id="language" className="h-9 rounded-md bg-transparent px-2 text-xs text-muted-foreground" value={locale} onChange={(event) => setLocale(event.target.value as "ru" | "en")}><option value="ru">RU</option><option value="en">EN</option></select>
            <Button className="h-9 w-9 px-0" variant="ghost" aria-label={t("settings.theme")} onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>{theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}</Button>
            <Button className="h-9 w-9 px-0" variant="ghost" aria-label={t("common.close")} onClick={() => logout.mutate()}><LogOut size={17} /></Button>
          </div>
        </header>
        {!backendOnline ? <div role="status" className="border-b border-destructive/30 bg-destructive/10 px-4 py-2 text-center text-xs text-destructive">{t("errors.backendUnavailable")}</div> : null}
        <div className="mx-auto max-w-[1600px] p-4 sm:p-6"><Outlet /></div>
      </main>
    </div>
  );
}
