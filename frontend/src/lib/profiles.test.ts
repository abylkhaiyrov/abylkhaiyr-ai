import { describe, expect, it } from "vitest";
import type { InferenceProfile, ProfileId, UserModel } from "@/lib/api";
import { hasProduction, pickProfile, preferredModel } from "@/lib/profiles";

const profile = (id: ProfileId, available = true, allowed = true): InferenceProfile => ({
  id, max_output_tokens: 1, context_budget: 1, generation_timeout_seconds: 1, tool_budget_seconds: 0,
  reasoning: { enabled: false, max_candidates: 1, verification_attempts: 0 }, limits: { max_total_new_tokens: 1, max_concurrent_generations: 1 },
  available, unavailable_reason: available ? null : "context_too_small", allowed,
});
const model = (id: string, quality_status: UserModel["quality_status"], available = true): UserModel => ({
  id, display_name: id, family: id, version: null, parameters: null, context_length: null, quality_status, capabilities: ["chat"], available, recommended_use: "chat",
});

describe("pickProfile", () => {
  it("keeps a usable preference", () => {
    expect(pickProfile([profile("lite"), profile("medium")], "lite")?.id).toBe("lite");
  });
  it("skips unavailable or disallowed profiles", () => {
    expect(pickProfile([profile("lite", false), profile("medium", true, false), profile("high")], "lite")?.id).toBe("high");
  });
  it("defaults to medium when there is no preference", () => {
    expect(pickProfile([profile("lite"), profile("medium")], null)?.id).toBe("medium");
  });
  it("returns null when nothing is usable", () => {
    expect(pickProfile([profile("lite", false)], null)).toBeNull();
    expect(pickProfile(undefined, "lite")).toBeNull();
  });
});

describe("preferredModel", () => {
  it("prefers the saved choice, then production", () => {
    const models = [model("a", "experimental"), model("b", "production")];
    expect(preferredModel(models, "a")?.id).toBe("a");
    expect(preferredModel(models, null)?.id).toBe("b");
  });
  it("ignores unavailable models and handles empty lists", () => {
    expect(preferredModel([model("a", "production", false)], "a")).toBeNull();
    expect(preferredModel([], null)).toBeNull();
  });
  it("detects missing production", () => {
    expect(hasProduction([model("a", "validated")])).toBe(false);
    expect(hasProduction([model("a", "production")])).toBe(true);
  });
});
