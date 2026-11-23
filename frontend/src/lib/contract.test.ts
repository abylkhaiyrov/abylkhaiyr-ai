import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import samples from "@/lib/contract-samples.json";
import wsSamples from "@/lib/ws-samples.json";
import { parseServerEvent } from "@/lib/events";

const calls: Record<string, () => Promise<unknown>> = {
  "admin GET /api/runtime": () => api.runtime(),
  "admin GET /api/catalog/runs": () => api.catalogRuns(),
  "admin GET /api/catalog/runs/{id}": () => api.catalogRun("run:demo_v3"),
  "admin GET /api/catalog/runs/{id} pipeline": () => api.catalogRun("pipeline:pipeline_tiny_v2"),
  "admin GET /api/runs": () => api.runs(),
  "admin GET /api/pipelines": () => api.pipelines(),
  "admin GET /api/experiments": () => api.experiments(),
  "admin GET /api/processes": () => api.processes(),
  "admin GET /api/processes/{id}/events": () => api.processEvents("proc-1"),
  "admin GET /api/models/registry": () => api.modelRegistry(),
  "admin GET /api/models/catalog": () => api.modelCatalog(),
  "admin GET /api/models/allowed": () => api.allowedModels(),
  "admin GET /api/processes/page": () => api.processPage({ group: "completed" }),
  "admin GET /api/experiments/page": () => api.experimentPage(),
  "admin GET /api/models/checkpoints": () => api.modelCheckpoints(),
  "admin GET /api/inference/profiles": () => api.inferenceProfiles(),
  "admin GET /api/evaluation/metrics": () => api.evaluationMetrics(),
  "admin GET /api/evaluation/groups": () => api.evaluationGroups(),
  "admin GET /api/evaluation/groups/{id}": () => api.evaluationGroup("group"),
  "admin GET /api/evaluation/compare": () => api.compareEvaluations("group", ["a", "b"]),
  "user GET /api/chat/models": () => api.userModels(),
  "user GET /api/chat/model": () => api.chatModel(),
  "user GET /api/inference/profiles": () => api.inferenceProfiles(),
};

const respond = (body: unknown) => vi.fn().mockResolvedValue(new Response(JSON.stringify(body), { headers: { "content-type": "application/json" } }));

afterEach(() => vi.unstubAllGlobals());

describe("backend samples are accepted by the client guards", () => {
  it("covers every sampled endpoint", () => {
    expect(Object.keys(samples).sort()).toEqual(Object.keys(calls).sort());
  });

  it.each(Object.keys(calls))("%s", async (name) => {
    vi.stubGlobal("fetch", respond((samples as Record<string, unknown>)[name]));
    await expect(calls[name]()).resolves.not.toThrow();
  });

  it("rejects a response that lost a required field", async () => {
    const broken = structuredClone(samples["admin GET /api/catalog/runs"]) as { items: { id?: string }[] };
    delete broken.items[0].id;
    vi.stubGlobal("fetch", respond(broken));
    await expect(api.catalogRuns()).rejects.toMatchObject({ code: "invalid_response" });
  });

  it("surfaces backend error details", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "model_not_allowed" }), { status: 403, headers: { "content-type": "application/json" } })));
    await expect(api.userModels()).rejects.toMatchObject({ code: "forbidden", detail: "model_not_allowed" });
  });
});

describe("websocket events", () => {
  it.each(Object.entries(wsSamples))("%s is understood by the client parser", (type, event) => {
    const parsed = parseServerEvent(JSON.stringify(event));
    expect(parsed).not.toBeNull();
    expect(parsed?.type).toBe(type);
  });
});
