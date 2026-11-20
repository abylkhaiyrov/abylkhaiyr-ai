import type { Model, ModelRegistry, RuntimeState } from "@/lib/api";

export function defaultModel(registry: ModelRegistry | undefined, runtime: RuntimeState | undefined): Model | null {
  if (!registry) return null;
  const byId = (id: string | null | undefined) => registry.models.find((model) => model.id === id && model.compatible) ?? null;
  const production = runtime?.production?.ready ? byId(runtime.production.checkpoint) : null;
  return production ?? byId(registry.roles.loaded) ?? byId(registry.roles.best_evaluated) ?? byId(registry.roles.latest_trained) ?? registry.models.find((model) => model.compatible) ?? null;
}

export function modelById(registry: ModelRegistry | undefined, id: string | null): Model | null {
  return registry?.models.find((model) => model.id === id) ?? null;
}
