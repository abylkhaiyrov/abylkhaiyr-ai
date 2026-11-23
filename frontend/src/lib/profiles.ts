import type { InferenceProfile, ProfileId, UserModel } from "@/lib/api";

export const PROFILE_ORDER: ProfileId[] = ["lite", "medium", "high", "very_high"];

export const profileUsable = (profile: InferenceProfile) => profile.available && profile.allowed;

export function pickProfile(profiles: InferenceProfile[] | undefined, preferred: ProfileId | null | undefined): InferenceProfile | null {
  const usable = (profiles ?? []).filter(profileUsable);
  return usable.find((item) => item.id === preferred) ?? usable.find((item) => item.id === "medium") ?? usable[0] ?? null;
}

export function preferredModel(models: UserModel[] | undefined, preferred: string | null | undefined): UserModel | null {
  const usable = (models ?? []).filter((item) => item.available);
  return usable.find((item) => item.id === preferred) ?? usable.find((item) => item.quality_status === "production") ?? usable[0] ?? null;
}

export const hasProduction = (models: Pick<UserModel, "quality_status">[]) => models.some((item) => item.quality_status === "production");
