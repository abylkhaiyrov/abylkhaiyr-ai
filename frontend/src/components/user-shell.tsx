import { Suspense } from "react";
import { Link, Outlet, useLocation, useMatch } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { ConnectionStatus } from "@/components/connection-status";
import { LoadingState } from "@/components/data-state";
import { ErrorBoundary } from "@/components/error-boundary";
import { HeaderActions } from "@/components/header-actions";
import { Toaster } from "@/components/toaster";
import { cn } from "@/lib/utils";

export function UserShell({ apiError, retryApi }: { apiError: Error | null; retryApi: () => void }) {
  const { t } = useTranslation();
  const chatRoute = useMatch("/chat");
  const { pathname } = useLocation();
  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-background">
      <a href="#content" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-[80] focus:rounded-md focus:bg-background focus:px-3 focus:py-1.5">{t("app.skip")}</a>
      <header className="flex h-12 shrink-0 items-center justify-between gap-2 border-b hairline px-3">
        <Link to="/chat" className="flex items-center gap-2.5 rounded-md px-1">
          <span className="grid h-6 w-6 place-items-center rounded-md bg-accent text-xs font-bold text-accent-foreground">A</span>
          <span className="hidden text-[13px] font-semibold tracking-tight sm:inline">{t("app.name")}</span>
        </Link>
        <HeaderActions />
      </header>
      <ConnectionStatus error={apiError} retry={retryApi} />
      <main id="content" tabIndex={-1} className={cn("min-h-0 flex-1 outline-none", chatRoute ? "overflow-hidden" : "overflow-y-auto")}>
        <ErrorBoundary key={pathname} page>
          <Suspense fallback={<LoadingState />}>
            {chatRoute ? <Outlet /> : <div className="mx-auto w-full max-w-3xl px-4 py-6"><Outlet /></div>}
          </Suspense>
        </ErrorBoundary>
      </main>
      <Toaster />
    </div>
  );
}
