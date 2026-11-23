import { useQuery } from "@tanstack/react-query";
import { useRole } from "@/hooks/use-auth";
import { api, type Metric } from "@/lib/api";

const LIVE = { staleTime: Infinity, refetchOnReconnect: false } as const;

export const useRuntime = () => useQuery({ queryKey: ["runtime"], queryFn: api.runtime, enabled: useRole() === "ADMIN", ...LIVE });
export const useSystem = () => useQuery({ queryKey: ["system"], queryFn: api.system, ...LIVE });
export const useModelRegistry = () => useQuery({ queryKey: ["models"], queryFn: api.modelRegistry, enabled: useRole() === "ADMIN", ...LIVE });
export const useModelCatalog = () => useQuery({ queryKey: ["model-catalog", "picker"], queryFn: () => api.modelCatalog({ limit: 20 }), select: (page) => page.items, enabled: useRole() === "ADMIN", staleTime: 30_000 });
export const useInferenceProfiles = () => useQuery({ queryKey: ["inference-profiles"], queryFn: api.inferenceProfiles, staleTime: 60_000 });
export const useUserModels = () => useQuery({ queryKey: ["user-models"], queryFn: api.userModels, enabled: useRole() === "USER", staleTime: 60_000 });
export const useSystemHistory = () =>
  useQuery({ queryKey: ["system-history"], queryFn: () => api.systemHistory(3600), ...LIVE });

export function useActiveMetrics(run: string | null, live: boolean) {
  return useQuery<Metric[]>({
    queryKey: ["metrics", run],
    queryFn: () => api.runMetrics(run ?? ""),
    enabled: run !== null,
    staleTime: live ? Infinity : 60_000,
  });
}
