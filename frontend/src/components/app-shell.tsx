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
  Menu,
  PanelLeftClose,
  PanelLeftOpen,
  Settings,
  Users,
  TerminalSquare,
  X,
  type LucideIcon,
  History as HistoryIcon,
} from "lucide-react";
import { Suspense, useState } from "react";
import { NavLink, Outlet, useLocation, useMatch } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { ConnectionStatus } from "@/components/connection-status";
import { ErrorBoundary } from "@/components/error-boundary";
import { LoadingState } from "@/components/data-state";
import { HeaderActions } from "@/components/header-actions";
import { Toaster } from "@/components/toaster";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useUiStore } from "@/store/ui-store";

type NavItem = readonly [string, string, LucideIcon];

const primary: NavItem[] = [
  ["/chat", "chat", Bot],
  ["/", "overview", LayoutDashboard],
  ["/training", "training", Activity],
  ["/monitoring", "monitoring", BarChart3],
];
const management: NavItem[] = [
  ["/models", "models", Boxes],
  ["/datasets", "datasets", Database],
  ["/experiments", "experiments", FlaskConical],
  ["/evaluation", "evaluation", ClipboardList],
  ["/history", "history", HistoryIcon],
  ["/processes", "processes", TerminalSquare],
  ["/logs", "logs", FileText],
  ["/users", "users", Users],
];

function NavGroup({ items, collapsed, close, title }: { items: NavItem[]; collapsed: boolean; close: () => void; title?: string }) {
  const { t } = useTranslation();
  return (
    <div className="grid gap-0.5">
      {title ? <p className={cn("label-caps px-2.5 pb-1 pt-4", collapsed && "sr-only")}>{title}</p> : null}
      {items.map(([to, label, Icon]) => (
        <NavLink
          key={to}
          to={to}
          end={to === "/"}
          onClick={close}
          aria-label={t(`nav.${label}`)}
          title={collapsed ? t(`nav.${label}`) : undefined}
          className={({ isActive }) =>
            cn(
              "flex h-8 items-center gap-2.5 rounded-md px-2.5 text-[13px] transition-colors",
              collapsed && "justify-center px-0",
              isActive ? "bg-muted font-medium text-foreground" : "text-muted-foreground hover:bg-muted/60 hover:text-foreground",
            )
          }
        >
          <Icon size={16} className="shrink-0" />
          {collapsed ? null : <span className="truncate">{t(`nav.${label}`)}</span>}
        </NavLink>
      ))}
    </div>
  );
}

function Navigation({ collapsed, close }: { collapsed: boolean; close: () => void }) {
  const { t } = useTranslation();
  return (
    <nav aria-label={t("app.name")} className="flex min-h-0 flex-1 flex-col overflow-y-auto px-2 pb-2">
      <NavGroup items={primary} collapsed={collapsed} close={close} />
      <NavGroup items={management} collapsed={collapsed} close={close} title={t("nav.more")} />
      <div className="mt-auto pt-3">
        <NavGroup items={[["/settings", "settings", Settings]]} collapsed={collapsed} close={close} />
      </div>
    </nav>
  );
}

function Brand({ collapsed }: { collapsed: boolean }) {
  const { t } = useTranslation();
  return (
    <div className={cn("flex h-12 shrink-0 items-center gap-2.5", collapsed ? "justify-center" : "px-4")}>
      <div className="grid h-6 w-6 shrink-0 place-items-center rounded-md bg-accent text-xs font-bold text-accent-foreground">A</div>
      {collapsed ? null : <span className="truncate text-[13px] font-semibold tracking-tight">{t("app.name")}</span>}
    </div>
  );
}

export function AppShell({ apiError, retryApi }: { apiError: Error | null; retryApi: () => void }) {
  const { t } = useTranslation();
  const chatRoute = useMatch("/chat");
  const { pathname } = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);
  const { sidebarCollapsed, toggleSidebar } = useUiStore();
  return (
    <div className="flex h-dvh overflow-hidden bg-background">
      <a href="#content" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-[80] focus:rounded-md focus:bg-background focus:px-3 focus:py-1.5">{t("app.skip")}</a>
      <aside className={cn("hidden shrink-0 flex-col border-r hairline transition-[width] duration-200 lg:flex", sidebarCollapsed ? "w-14" : "w-56")}>
        <Brand collapsed={sidebarCollapsed} />
        <Navigation collapsed={sidebarCollapsed} close={() => undefined} />
      </aside>
      {mobileOpen ? (
        <div className="fixed inset-0 z-50 bg-black/50 lg:hidden" onClick={() => setMobileOpen(false)}>
          <aside className="flex h-full w-64 flex-col bg-background shadow-xl" onClick={(event) => event.stopPropagation()}>
            <div className="flex h-12 items-center justify-between pr-2"><Brand collapsed={false} /><Button variant="ghost" size="icon" aria-label={t("common.close")} onClick={() => setMobileOpen(false)}><X size={16} /></Button></div>
            <Navigation collapsed={false} close={() => setMobileOpen(false)} />
          </aside>
        </div>
      ) : null}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-12 shrink-0 items-center justify-between gap-2 border-b hairline px-3">
          <div className="flex items-center gap-1.5">
            <Button className="lg:hidden" variant="ghost" size="icon" aria-label={t("app.openMenu")} onClick={() => setMobileOpen(true)}><Menu size={16} /></Button>
            <Button className="hidden lg:inline-flex" variant="ghost" size="icon" aria-label={t(sidebarCollapsed ? "app.expandSidebar" : "app.collapseSidebar")} onClick={toggleSidebar}>
              {sidebarCollapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
            </Button>
          </div>
          <HeaderActions />
        </header>
        <ConnectionStatus error={apiError} retry={retryApi} />
        <main id="content" tabIndex={-1} className={cn("min-h-0 flex-1 outline-none", chatRoute ? "overflow-hidden" : "overflow-y-auto")}>
          <ErrorBoundary key={pathname} page>
            <Suspense fallback={<LoadingState />}>
              {chatRoute ? <Outlet /> : <div className="mx-auto w-full max-w-[1440px] px-4 py-5 lg:px-7"><Outlet /></div>}
            </Suspense>
          </ErrorBoundary>
        </main>
      </div>
      <Toaster />
    </div>
  );
}
