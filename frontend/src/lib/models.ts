import type { Model, ModelRegistry, RuntimeState } from "@/lib/api";

export function defaultModel(registry: ModelRegistry | undefined, runtime: RuntimeState | undefined): Model | null {
  if (!registry) return null;
  const byId = (id: string | null | undefined) => registry.models.find((model) => model.id === id && model.compatible) ?? null;
  const production = runtime?.production?.ready ? byId(runtime.production.checkpoint) : null;
  const chatModels = registry.models.filter((model) => model.compatible && model.available && !model.quality_failed && ["sft", "dpo"].includes(model.stage));
  const loadedChat = chatModels.find((model) => model.id === registry.roles.loaded);
  const evaluatedChat = chatModels.find((model) => model.id === registry.roles.best_evaluated);
  const latestChat = [...chatModels].sort((a, b) => (b.created_at ?? "").localeCompare(a.created_at ?? ""))[0];
  return production ?? loadedChat ?? evaluatedChat ?? latestChat ?? byId(registry.roles.loaded) ?? byId(registry.roles.best_evaluated) ?? byId(registry.roles.latest_trained) ?? registry.models.find((model) => model.compatible) ?? null;
}

export function modelById(registry: ModelRegistry | undefined, id: string | null): Model | null {
  return registry?.models.find((model) => model.id === id) ?? null;
}

export function modelAvailabilityKey(model: Model): string | null {
  if (!model.compatible) {
    return model.compatibility_issues[0] ?? "incompatible_model";
  }
  if (!model.available && model.availability_reason) {
    return model.availability_reason.split(" ")[0];
  }
  return null;
}
