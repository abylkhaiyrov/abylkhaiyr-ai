import { body, http } from "@/lib/http";
import { anything, arrayOf, boolean, listOf, nullable, number, record, shape, string } from "@/lib/schema";

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
  note_code?: "sft_perplexity_regression";
  note_values?: { sft: number; limit: number; pretrain: number };
  history?: { status: StageStatus; at: string }[];
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
  trained: boolean;
  evaluated: boolean;
  quality_failed: boolean;
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
  context_messages?: number;
};

export type StudioEvent = { process_id: string; level: string; message: string; created_at: string };
export type Dataset = { name: string; path: string; size_bytes: number; manifest: Record<string, unknown> };
export type Experiment = { name: string; results: Metric; [key: string]: unknown };
export type Evaluation = {
  path: string;
  experiment: string;
  dataset: string | null;
  stage: string | null;
  model: string | null;
  checkpoint: string | null;
  results: Metric;
};

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
  variants: MessageVariant[];
  status: string;
  created_at: string;
};
export type MessageVariant = {
  content: string;
  model_id: string | null;
  checkpoint_id: string | null;
  output_tokens: number | null;
  generation_ms: number | null;
  tokens_per_second: number | null;
  status: string;
  created_at: string;
};
export type Chat = ChatSummary & { messages: ChatMessage[] };

const named = shape<{ name: string }>({ name: string });
const me = shape<{ username: string; requires_password_change: boolean }>({ username: string, requires_password_change: boolean });
const pipelineGuard = shape<Pipeline>({ id: string, stages: arrayOf(shape<PipelineStage>({ key: string, status: string })), running: boolean });
const runtimeGuard = shape<RuntimeState>({ generated_at: string, pipeline: nullable(pipelineGuard), compute: shape({ busy: boolean }), models: record(anything), events: arrayOf(anything) });
const registryGuard = shape<ModelRegistry>({ roles: record(anything), models: arrayOf(shape<Model>({ id: string, roles: arrayOf(string) })) });
const systemGuard = shape<SystemSnapshot>({ recorded_at: string, cpu: shape({ percent: number }), memory: shape({ total_bytes: number }), gpu: shape({ device: string }), disk: shape({ total_bytes: number }) });
const chatSummary = shape<ChatSummary>({ id: string, title: string, pinned: boolean, updated_at: string });
const chatGuard = shape<Chat>({ id: string, title: string, messages: arrayOf(shape<ChatMessage>({ id: string, role: string, content: string, variants: arrayOf(anything), reasoning_trace: arrayOf(anything) })) });
const rows = listOf<Metric>(shape({ step: number }));

export const api = {
  login: (username: string, password: string) => http<{ username: string; requires_password_change: boolean }>("/auth/login", body("POST", { username, password }), me, { session: false }),
  me: () => http<{ username: string; requires_password_change: boolean }>("/auth/me", {}, me, { session: false }),
  logout: () => http<void>("/auth/logout", body("POST"), undefined, { session: false }),
  changePassword: (currentPassword: string, newPassword: string) => http<void>("/auth/password", body("PUT", { current_password: currentPassword, new_password: newPassword })),
  runtime: () => http<RuntimeState>("/runtime", {}, runtimeGuard),
  system: () => http<SystemSnapshot>("/system", {}, systemGuard),
  systemHistory: (seconds: number) => http<SystemSnapshot[]>(`/system/history?seconds=${seconds}`, {}, listOf<SystemSnapshot>(systemGuard)),
  processes: () => http<Process[]>("/processes", {}, listOf<Process>(shape({ id: string, status: string }))),
  startTraining: (configPath: string, stage?: string) => http<Process>("/training", body("POST", { config_path: configPath, stage })),
  pauseTraining: (id: string) => http<Process>(`/training/${id}/pause`, body("POST")),
  stopTraining: (id: string) => http<Process>(`/training/${id}/stop`, body("POST")),
  stopPipeline: () => http<{ pid: number }>("/pipeline/stop", body("POST")),
  trainingConfigs: () => http<{ path: string; name: string; stage: string }[]>("/training/configs", {}, listOf(shape({ path: string, name: string, stage: string }))),
  runs: () => http<Run[]>("/runs", {}, listOf<Run>(shape({ name: string, status: string, stage: string }))),
  run: (name: string) => http<RunDetail>(`/runs/${encodeURIComponent(name)}`, {}, shape<RunDetail>({ name: string, checkpoints: arrayOf(anything), evaluations: arrayOf(anything) })),
  runMetrics: (name: string, maxPoints = 600) => http<Metric[]>(`/runs/${encodeURIComponent(name)}/metrics?max_points=${maxPoints}`, {}, rows),
  runLogs: (name: string, limit = 300) => http<string[]>(`/runs/${encodeURIComponent(name)}/logs?limit=${limit}`, {}, listOf<string>(string)),
  modelRegistry: () => http<ModelRegistry>("/models/registry", {}, registryGuard),
  switchModel: (checkpointId: string, device?: string) => http<Model>("/models/switch", body("POST", { checkpoint_id: checkpointId, device })),
  unloadModel: () => http<{ loaded: boolean }>("/models/unload", body("POST")),
  datasets: () => http<Dataset[]>("/datasets", {}, listOf<Dataset>(shape({ name: string, size_bytes: number }))),
  experiments: () => http<Experiment[]>("/experiments", {}, listOf<Experiment>(named)),
  pipelines: () => http<Pipeline[]>("/pipelines", {}, listOf(pipelineGuard)),
  evaluations: () => http<Evaluation[]>("/evaluation", {}, listOf<Evaluation>(shape({ path: string, experiment: string, dataset: nullable(string), stage: nullable(string), model: nullable(string), checkpoint: nullable(string), results: record(number) }))),
  logs: (category?: string) => http<{ items: { category: string; source: string; message: string }[]; total: number }>(`/logs${category ? `?category=${category}` : ""}`, {}, shape({ items: arrayOf(shape({ message: string })), total: number })),
  chats: (query?: string) => http<ChatSummary[]>(`/chats${query ? `?query=${encodeURIComponent(query)}` : ""}`, {}, listOf<ChatSummary>(chatSummary)),
  chat: (id: string) => http<Chat>(`/chats/${id}`, {}, chatGuard),
  createChat: (payload?: { title?: string; model_name?: string; checkpoint?: string; system_prompt?: string; generation_settings?: GenerationSettings }) => http<ChatSummary>("/chats", body("POST", payload ?? {}), chatSummary),
  updateChat: (id: string, payload: { title?: string; model_name?: string; checkpoint?: string; system_prompt?: string; generation_settings?: GenerationSettings; pinned?: boolean }) => http<ChatSummary>(`/chats/${id}`, body("PATCH", payload), chatSummary),
  deleteChat: (id: string) => http<void>(`/chats/${id}`, body("DELETE")),
  selectVariant: (chatId: string, messageId: string, index: number) => http<ChatMessage>(`/chats/${chatId}/messages/${messageId}/variants/${index}`, body("POST")),
  exportChatUrl: (id: string, format: "md" | "json") => `/api/chats/${id}/export?format=${format}`,
};
