import { lazy, Suspense, useEffect, useSyncExternalStore } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { onlineManager, QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";

import { AppShell } from "@/components/app-shell";
import { ErrorState, LoadingState } from "@/components/data-state";
import { ErrorBoundary } from "@/components/error-boundary";
import { StudioEventsProvider } from "@/hooks/studio-events-provider";
import { api } from "@/lib/api";
import { ApiError, setUnauthorizedHandler } from "@/lib/http";

const LoginPage = lazy(() => import("@/pages/login-page").then((module) => ({ default: module.LoginPage })));
const DashboardPage = lazy(() => import("@/pages/dashboard-page").then((module) => ({ default: module.DashboardPage })));
const TrainingPage = lazy(() => import("@/pages/training-page").then((module) => ({ default: module.TrainingPage })));
const MonitoringPage = lazy(() => import("@/pages/monitoring-page").then((module) => ({ default: module.MonitoringPage })));
const ChatPage = lazy(() => import("@/pages/chat-page").then((module) => ({ default: module.ChatPage })));
const ProcessesPage = lazy(() => import("@/pages/resource-pages").then((module) => ({ default: module.ProcessesPage })));
const ModelsPage = lazy(() => import("@/pages/resource-pages").then((module) => ({ default: module.ModelsPage })));
const DatasetsPage = lazy(() => import("@/pages/resource-pages").then((module) => ({ default: module.DatasetsPage })));
const ExperimentsPage = lazy(() => import("@/pages/resource-pages").then((module) => ({ default: module.ExperimentsPage })));
const EvaluationPage = lazy(() => import("@/pages/evaluation-page").then((module) => ({ default: module.EvaluationPage })));
const LogsPage = lazy(() => import("@/pages/resource-pages").then((module) => ({ default: module.LogsPage })));
const SettingsPage = lazy(() => import("@/pages/settings-page").then((module) => ({ default: module.SettingsPage })));

const client = new QueryClient({
  defaultOptions: {
    queries: {
      retry: (count, error) => count < 1 && error instanceof ApiError && error.status >= 500 && error.code !== "unavailable",
      refetchOnWindowFocus: false,
      staleTime: 15_000,
    },
  },
});

setUnauthorizedHandler(() => {
  void client.invalidateQueries({ queryKey: ["me"] });
});

const watchNetwork = (listener: () => void) => onlineManager.subscribe(listener);
const isOnline = () => onlineManager.isOnline();

function ProtectedApp() {
  const location = useLocation();
  const online = useSyncExternalStore(watchNetwork, isOnline);
  const auth = useQuery({
    queryKey: ["me"],
    queryFn: ({ signal }) => api.me(signal),
    retry: false,
    staleTime: 15_000,
    refetchInterval: (query) => {
      const error = query.state.error;
      if (error instanceof ApiError && ["unauthorized", "incompatible"].includes(error.code)) return false;
      return error ? 10_000 : 30_000;
    },
  });
  const unauthorized = auth.error instanceof ApiError && auth.error.code === "unauthorized";
  useEffect(() => {
    if (unauthorized) {
      client.removeQueries({ predicate: (query) => query.queryKey[0] !== "me" });
      client.getMutationCache().clear();
    }
  }, [unauthorized]);
  if (unauthorized) {
    return <Navigate to="/login" replace state={{ sessionExpired: auth.data !== undefined, returnTo: location.pathname + location.search }} />;
  }
  if (online && auth.isPending && !auth.isPaused) return <LoadingState />;
  const error = auth.error ?? (!online || auth.isPaused ? new ApiError(0, "offline") : null);
  const retry = () => void auth.refetch();
  if (!auth.data || (error instanceof ApiError && error.code === "incompatible")) {
    return <main className="mx-auto max-w-xl p-6"><ErrorState error={error ?? new ApiError(0, "network")} retry={retry} /></main>;
  }
  return (
    <StudioEventsProvider enabled={!error}>
      <Routes>
        <Route element={<AppShell apiError={error} retryApi={retry} />}>
          <Route index element={<DashboardPage />} />
          <Route path="training" element={<TrainingPage />} />
          <Route path="monitoring" element={<MonitoringPage />} />
          <Route path="processes" element={<ProcessesPage />} />
          <Route path="chat" element={<ChatPage />} />
          <Route path="models" element={<ModelsPage />} />
          <Route path="datasets" element={<DatasetsPage />} />
          <Route path="experiments" element={<ExperimentsPage />} />
          <Route path="evaluation" element={<EvaluationPage />} />
          <Route path="logs" element={<LogsPage />} />
          <Route path="settings" element={<SettingsPage />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </StudioEventsProvider>
  );
}

export function App() {
  return (
    <QueryClientProvider client={client}>
      <ErrorBoundary>
        <Suspense fallback={<LoadingState />}>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/*" element={<ProtectedApp />} />
          </Routes>
        </Suspense>
      </ErrorBoundary>
    </QueryClientProvider>
  );
}
