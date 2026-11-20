import { csrfToken } from "@/lib/utils";

export type Metric = Record<string, number>;
export type StageStatus = "pending" | "running" | "completed" | "skipped" | "failed" | "interrupted";
export type StageKind = "data" | "tokenize" | "train" | "eval" | "select" | "export";

export type StageKey =
  | "data_preparation"
  | "tokenization"
  | "pretrain"
  | "eval_pretrain"
  | "select_best_pretrain"
  | "sft"
  | "eval_sft"
  | "dpo"
  | "eval_final"
  | "export_release";

export type StageDetail = {
  phase?: string;
  checkpoint?: string;
  checkpoint_index?: number;
  checkpoint_total?: number;
  benchmark?: string;
  benchmark_index?: number;
  benchmark_total?: number;
  items_done?: number;
  items_total?: number | null;
  items_remaining?: number | null;
  tasks_done?: number;
  tasks_per_second?: number;
  rate?: number | null;
  interim?: Record<string, Record<string, number>>;
  validation_done?: number;
  validation_total?: number;
};

export type PipelineStage = {
  key: StageKey;
  kind: StageKind;
  status: StageStatus;
  started_at: string | null;
  finished_at: string | null;
  duration_seconds: number | null;
  done: number | null;
  total: number | null;
  unit: string | null;
  percent: number | null;
  indeterminate: boolean;
  eta_seconds: number | null;
  phase: string | null;
  detail: StageDetail;
  results: Record<string, number | string | null>;
  error: string | null;
  note: string | null;
  output_dir: string | null;
  checkpoint_dir: string | null;
};

export type Pipeline = {
  id: string;
  state_dir: string;
  running: boolean;
  pid: number | null;
  memory_bytes: number | null;
  cpu_percent: number | null;
  model_name: string;
  started_at: string | null;
  updated_at: string | null;
  stages: PipelineStage[];
  current_stage: StageKey | null;
  next_stage: StageKey | null;
  last_result_stage: StageKey | null;
  overall: { finished: number; total: number; percent: number; elapsed_seconds: number | null };
  artifacts: Record<string, string>;
  checkpoint: string | null;
};

export type ModelRole = "production" | "best_evaluated" | "latest_trained" | "loaded";

export type RoleModel = {
  id: string;
  display_name: string;
  step: number | null;
  stage: string;
  verified: boolean;
  device: string | null;
  created_at: string;
};

export type RuntimeState = {
  generated_at: string;
  pipeline: Pipeline | null;
  compute: { busy: boolean; holder: { kind: string; pid: number | null; stage: string | null } | null };
  models: Record<ModelRole, RoleModel | null>;
  production: { checkpoint: string; stage: string | null; ready: boolean; pass_rate: number | null; limitations: string[] } | null;
  inference_model: string | null;
  events: StudioEvent[];
};

export type SystemSnapshot = {
  recorded_at: string;
  cpu: { percent: number; per_core_percent: number[]; logical_cores: number | null; frequency_mhz: number | null };
  memory: { total_bytes: number; used_bytes: number; available_bytes: number; python_process_bytes: number; swap_total_bytes: number; swap_used_bytes: number };
  gpu: { device: string; allocated_bytes: number | null; driver_allocated_bytes: number | null; peak_bytes: number | null };
  disk: { total_bytes: number; used_bytes: number; free_bytes: number; datasets_bytes: number; checkpoints_bytes: number; database_bytes: number; logs_bytes: number };
};

export type Model = {
  id: string;
  model_id: string;
  checkpoint_id: string;
  name: string;
  display_name: string;
  step?: number | null;
  stage: string;
  compatible: boolean;
  compatibility_issues: string[];
  available: boolean;
  availability_reason: string | null;
  loaded: boolean;
  device: string | null;
  created_at?: string;
  weights_bytes?: number | null;
  parameters: number | null;
  vocab_size: number | null;
  context_length: number | null;
  roles: ModelRole[];
  verified: boolean;
  verified_stage: string | null;
  metrics?: Metric;
};

export type ModelRegistry = { roles: Record<ModelRole, string | null>; models: Model[] };

export type Process = {
  id: string;
  pid: number | null;
  kind: string;
  status: string;
  command: string;
  config_path: string | null;
  output_dir: string | null;
  started_at: string;
  completed_at: string | null;
  exit_code: number | null;
  error_reason: string | null;
  cpu_percent: number | null;
  memory_bytes: number | null;
  managed?: boolean;
  model_name?: string;
};

export type ReasoningStageEvent = { stage: string; status: string; data: Record<string, unknown>; timestamp?: number };

export type GenerationSettings = {
  mode?: "standard" | "reasoning" | "precise" | "creative";
  temperature?: number;
  top_p?: number;
  top_k?: number;
  max_tokens?: number;
  repetition_penalty?: number;
};

export type StudioEvent = { process_id: string; level: string; message: string; created_at: string };
export type Dataset = { name: string; path: string; size_bytes: number; manifest: Record<string, unknown> };
export type Experiment = { name: string; results: Metric; [key: string]: unknown };
export type Evaluation = { path: string; results: Metric };

export type Run = {
  name: string;
  output_dir: string;
  stage: "pretrain" | "sft" | "dpo";
  status: "running" | "completed" | "stopped";
  step: number | null;
  max_steps: number | null;
  percent: number | null;
  train_loss: number | null;
  validation_loss: number | null;
  validation_perplexity: number | null;
  best_validation_loss: number | null;
  first_train_loss: number | null;
  tokens_seen: number | null;
  config_path: string | null;
  started_at: string | null;
  updated_at: string | null;
};

export type RunCheckpoint = { id: string; label: string; step: number | null; tokens_seen: number | null; created_at: string };
export type RunDetail = Run & {
  checkpoints: RunCheckpoint[];
  config: { path: string; text: string; parsed: Record<string, unknown> } | null;
  evaluations: { name: string; checkpoint: string; results: Metric }[];
};

export type ChatSummary = {
  id: string;
  title: string;
  model_name: string | null;
  checkpoint: string | null;
  system_prompt: string;
  pinned: boolean;
  generation_settings: GenerationSettings;
  created_at: string;
  updated_at: string;
};

export type ChatMessage = {
  id: string;
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  input_tokens: number | null;
  output_tokens: number | null;
  generation_ms: number | null;
  tokens_per_second: number | null;
  model_id: string | null;
  checkpoint_id: string | null;
  generation_params: Record<string, unknown> | null;
  reasoning_trace: ReasoningStageEvent[];
  status: string;
  created_at: string;
};
export type Chat = ChatSummary & { messages: ChatMessage[] };

export class ApiError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body) headers.set("Content-Type", "application/json");
  if (init.method && !["GET", "HEAD"].includes(init.method)) {
    const token = csrfToken();
    if (token) headers.set("X-CSRF-Token", token);
  }
  const response = await fetch(`/api${path}`, { ...init, headers, credentials: "same-origin" });
  if (response.status === 204) return undefined as T;
  if (!response.ok) {
    const payload: unknown = await response.json().catch(() => ({}));
    const detail = typeof payload === "object" && payload !== null && "detail" in payload ? String(payload.detail) : response.statusText;
    throw new ApiError(response.status, detail);
  }
  return response.json() as Promise<T>;
}

const json = (method: string, body?: unknown): RequestInit => ({ method, body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  login: (username: string, password: string) => request<{ username: string; requires_password_change: boolean }>("/auth/login", json("POST", { username, password })),
  me: () => request<{ username: string; requires_password_change: boolean }>("/auth/me"),
  logout: () => request<void>("/auth/logout", json("POST")),
  changePassword: (currentPassword: string, newPassword: string) => request<void>("/auth/password", json("PUT", { current_password: currentPassword, new_password: newPassword })),
  runtime: () => request<RuntimeState>("/runtime"),
  system: () => request<SystemSnapshot>("/system"),
  systemHistory: (seconds: number) => request<SystemSnapshot[]>(`/system/history?seconds=${seconds}`),
  processes: () => request<Process[]>("/processes"),
  startTraining: (configPath: string, stage?: string) => request<Process>("/training", json("POST", { config_path: configPath, stage })),
  pauseTraining: (id: string) => request<Process>(`/training/${id}/pause`, json("POST")),
  stopTraining: (id: string) => request<Process>(`/training/${id}/stop`, json("POST")),
  stopPipeline: () => request<{ pid: number }>("/pipeline/stop", json("POST")),
  trainingConfigs: () => request<{ path: string; name: string; stage: string }[]>("/training/configs"),
  runs: () => request<Run[]>("/runs"),
  run: (name: string) => request<RunDetail>(`/runs/${encodeURIComponent(name)}`),
  runMetrics: (name: string, maxPoints = 600) => request<Metric[]>(`/runs/${encodeURIComponent(name)}/metrics?max_points=${maxPoints}`),
  runLogs: (name: string, limit = 300) => request<string[]>(`/runs/${encodeURIComponent(name)}/logs?limit=${limit}`),
  modelRegistry: () => request<ModelRegistry>("/models/registry"),
  switchModel: (checkpointId: string, device?: string) => request<Model>("/models/switch", json("POST", { checkpoint_id: checkpointId, device })),
  unloadModel: () => request<{ loaded: boolean }>("/models/unload", json("POST")),
  datasets: () => request<Dataset[]>("/datasets"),
  experiments: () => request<Experiment[]>("/experiments"),
  evaluations: () => request<Evaluation[]>("/evaluation"),
  logs: (category?: string) => request<{ items: { category: string; source: string; message: string }[]; total: number }>(`/logs${category ? `?category=${category}` : ""}`),
  chats: (query?: string) => request<ChatSummary[]>(`/chats${query ? `?query=${encodeURIComponent(query)}` : ""}`),
  chat: (id: string) => request<Chat>(`/chats/${id}`),
  createChat: (payload?: { title?: string; model_name?: string; checkpoint?: string; system_prompt?: string; generation_settings?: GenerationSettings }) => request<ChatSummary>("/chats", json("POST", payload ?? {})),
  updateChat: (id: string, payload: { title?: string; model_name?: string; checkpoint?: string; system_prompt?: string; generation_settings?: GenerationSettings; pinned?: boolean }) => request<ChatSummary>(`/chats/${id}`, json("PATCH", payload)),
  deleteChat: (id: string) => request<void>(`/chats/${id}`, json("DELETE")),
  exportChatUrl: (id: string, format: "md" | "json") => `/api/chats/${id}/export?format=${format}`,
};
