import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";

import { AppShell } from "@/components/app-shell";
import { LoadingState } from "@/components/data-state";
import { StudioEventsProvider } from "@/hooks/studio-events-provider";
import i18n from "@/i18n";
import { api } from "@/lib/api";
import { setUnauthorizedHandler } from "@/lib/http";
import { useToastStore } from "@/store/toast-store";

const LoginPage = lazy(() => import("@/pages/login-page").then((module) => ({ default: module.LoginPage })));
const DashboardPage = lazy(() => import("@/pages/dashboard-page").then((module) => ({ default: module.DashboardPage })));
const TrainingPage = lazy(() => import("@/pages/training-page").then((module) => ({ default: module.TrainingPage })));
const MonitoringPage = lazy(() => import("@/pages/monitoring-page").then((module) => ({ default: module.MonitoringPage })));
const ChatPage = lazy(() => import("@/pages/chat-page").then((module) => ({ default: module.ChatPage })));
const ProcessesPage = lazy(() => import("@/pages/resource-pages").then((module) => ({ default: module.ProcessesPage })));
const ModelsPage = lazy(() => import("@/pages/resource-pages").then((module) => ({ default: module.ModelsPage })));
const DatasetsPage = lazy(() => import("@/pages/resource-pages").then((module) => ({ default: module.DatasetsPage })));
const ExperimentsPage = lazy(() => import("@/pages/resource-pages").then((module) => ({ default: module.ExperimentsPage })));
const EvaluationPage = lazy(() => import("@/pages/resource-pages").then((module) => ({ default: module.EvaluationPage })));
const LogsPage = lazy(() => import("@/pages/resource-pages").then((module) => ({ default: module.LogsPage })));
const SettingsPage = lazy(() => import("@/pages/settings-page").then((module) => ({ default: module.SettingsPage })));

const client = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 15_000,
    },
  },
});

let sessionNoticeShown = false;
setUnauthorizedHandler(() => {
  if (!sessionNoticeShown) {
    sessionNoticeShown = true;
    useToastStore.getState().push("info", i18n.t("auth.sessionExpired"));
  }
  void client.invalidateQueries({ queryKey: ["me"] });
});

function ProtectedApp() {
  const auth = useQuery({ queryKey: ["me"], queryFn: api.me, retry: false, staleTime: Infinity });
  if (auth.isSuccess) sessionNoticeShown = false;
  if (auth.isPending) return <LoadingState />;
  if (auth.isError) return <Navigate to="/login" replace />;
  return (
    <StudioEventsProvider><Suspense fallback={<LoadingState />}>
      <Routes>
        <Route element={<AppShell />}>
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
    </Suspense></StudioEventsProvider>
  );
}

export function App() {
  return (
    <QueryClientProvider client={client}>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/*" element={<ProtectedApp />} />
      </Routes>
    </QueryClientProvider>
  );
}
