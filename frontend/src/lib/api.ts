import { body, http } from "@/lib/http";
import { anything, arrayOf, boolean, listOf, nullable, number, record, shape, string } from "@/lib/schema";

export type Role = "ADMIN" | "USER";
export type User = {
  id: string;
  first_name: string;
  last_name: string;
  email: string | null;
  username: string;
  role: Role;
  is_active: boolean;
  requires_password_change: boolean;
  created_at: string;
  last_login_at: string | null;
  last_active_at: string | null;
};
export type UserPage = { items: User[]; total: number };
export type UserSession = { id: number; created_at: string; last_seen_at: string | null; user_agent: string | null; current: boolean };
export type NotificationType = "GENERATION_COMPLETED" | "GENERATION_FAILED" | "GENERATION_CANCELLED" | "SYSTEM" | "ACCOUNT";
export type AppNotification = {
  id: string;
  type: NotificationType;
  title: string;
  message: string;
  chat_id: string | null;
  generation_id: string | null;
  is_read: boolean;
  read_at: string | null;
  created_at: string;
};
export type NotificationPage = { items: AppNotification[]; total: number; unread: number };
export type NotificationQuery = { unread: boolean; type: NotificationType | null; limit: number; offset: number };
export type ChatModelInfo = { id: string; display_name: string; stage: string; context_length: number | null; device: string | null };
export type RegisterPayload = { firstName: string; lastName: string; email: string; username: string; password: string };
export type ProfilePayload = { firstName: string; lastName: string; email: string; username: string };
export type UserChanges = Partial<{ firstName: string; lastName: string; email: string; username: string; role: Role; isActive: boolean }>;

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

export type RunStatus = "running" | "completed" | "stopped" | "failed";

export type RunEntry = {
  id: string;
  kind: "pipeline" | "training";
  name: string;
  model_name: string;
  version: string | null;
  stage: string;
  status: RunStatus;
  origin: "pipeline_state" | "metrics";
  created_at: string | null;
  updated_at: string | null;
  step: number | null;
  max_steps: number | null;
  percent: number | null;
  current_stage: string | null;
  output_dir: string | null;
  state_dir: string | null;
  config_path: string | null;
  checkpoint: string | null;
  failure_reason: string | null;
  validation_loss: number | null;
  best_validation_loss: number | null;
};

export type RunsSummary = { active: RunEntry | null; latest: RunEntry | null; completed_count: number; total: number };

export type RunQuery = { status?: RunStatus; stage?: string; version?: string; q?: string; sort?: "newest" | "oldest"; limit?: number; offset?: number };

export type RunPage = { items: RunEntry[]; total: number };

export type CheckpointStatus = "production" | "validated" | "candidate" | "experimental" | "rejected";

export type CatalogModel = {
  id: string;
  display_name: string;
  family: string;
  version: string | null;
  parameters: number | null;
  context_length: number | null;
  quality_status: CheckpointStatus;
  stage: string;
  capabilities: string[];
  available: boolean;
  availability_reason: string | null;
  recommended_use: "chat" | "research";
  representative_checkpoint: string;
  checkpoint_count: number;
  allowed: boolean;
};

export type UserModel = Pick<CatalogModel, "id" | "display_name" | "family" | "version" | "parameters" | "context_length" | "quality_status" | "capabilities" | "available" | "recommended_use">;

export type RuntimeState = {
  generated_at: string;
  pipeline: Pipeline | null;
  runs: RunsSummary;
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
  status: CheckpointStatus;
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
  duration_seconds?: number | null;
  exit_code: number | null;
  error_reason: string | null;
  cpu_percent: number | null;
  memory_bytes: number | null;
  managed?: boolean;
  model_name?: string;
};

export type ReasoningStageEvent = { stage: string; status: string; data: Record<string, unknown>; timestamp?: number };

export type ProfileId = "lite" | "medium" | "high" | "very_high";

export type InferenceProfile = {
  id: ProfileId;
  max_output_tokens: number;
  context_budget: number;
  generation_timeout_seconds: number;
  tool_budget_seconds: number;
  reasoning: { enabled: boolean; max_candidates: number; verification_attempts: number };
  limits: { max_total_new_tokens: number; max_concurrent_generations: number };
  available: boolean;
  unavailable_reason: string | null;
  allowed: boolean;
};

export type GenerationSettings = {
  profile?: ProfileId | null;
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
export type MetricDefinition = { key: string; section: "language" | "conversation" | "reasoning" | "code" | "multilingual" | "efficiency"; direction: "higher" | "lower"; unit: "ratio" | "number" | "seconds" | "gigabytes" | "rate"; language: string | null; tools: boolean | null };
export type GateStatus = "passed" | "failed" | "not_verified";
export type QualityGate = { id: string; status: GateStatus; value: number | string | null; threshold: number | string };
export type EvaluationGroupSummary = { id: string; experiment: string; dataset: string | null; methodology: string | null; split: string | null; checkpoints: number; evaluated_at: string | null; metric_count: number; gate_status: GateStatus };
export type EvaluationRow = { checkpoint: string | null; model: string | null; stage: string | null; evaluated_at: string | null; sources: string[]; results: Record<string, number>; gates: QualityGate[]; gate_status: GateStatus };
export type EvaluationGroup = EvaluationGroupSummary & { rows: EvaluationRow[] };
export type ComparisonEntry = { value: number | null; delta: number | null; relative: number | null; improved: boolean | null; ci: [number, number] | null; samples: number | null };
export type Comparison = {
  group: Pick<EvaluationGroup, "id" | "experiment" | "dataset" | "methodology" | "split">;
  checkpoints: Pick<EvaluationRow, "checkpoint" | "model" | "stage" | "gates" | "gate_status" | "evaluated_at">[];
  metrics: { key: string; direction: "higher" | "lower"; section: MetricDefinition["section"]; entries: ComparisonEntry[] }[];
  caveats: string[];
  verdict: { status: "leader" | "insufficient_data" | "inconclusive"; checkpoint: string | null; reason: string | null };
};
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
  archived: boolean;
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
export type Chat = ChatSummary & { messages: ChatMessage[]; total_messages: number };

const named = shape<{ name: string }>({ name: string });
const me = shape<User>({ id: string, username: string, role: string, is_active: boolean, requires_password_change: boolean });
const userPage = shape<UserPage>({ items: arrayOf(me), total: number });
const notification = shape<AppNotification>({ id: string, type: string, title: string, is_read: boolean });
const notificationPage = shape<NotificationPage>({ items: arrayOf(notification), total: number, unread: number });
const pipelineGuard = shape<Pipeline>({ id: string, stages: arrayOf(shape<PipelineStage>({ key: string, status: string })), running: boolean });
const runEntry = shape<RunEntry>({ id: string, kind: string, name: string, status: string, updated_at: nullable(string) });
const runsSummary = shape<RunsSummary>({ active: nullable(runEntry), latest: nullable(runEntry), completed_count: number, total: number });
const catalogModel = shape<CatalogModel>({ id: string, display_name: string, quality_status: string, representative_checkpoint: string, allowed: boolean, available: boolean });
const userModel = shape<UserModel>({ id: string, display_name: string, quality_status: string });
const runtimeGuard = shape<RuntimeState>({ generated_at: string, pipeline: nullable(pipelineGuard), runs: runsSummary, compute: shape({ busy: boolean }), models: record(anything), events: arrayOf(anything) });
const registryGuard = shape<ModelRegistry>({ roles: record(anything), models: arrayOf(shape<Model>({ id: string, roles: arrayOf(string), status: string })) });
const systemGuard = shape<SystemSnapshot>({ recorded_at: string, cpu: shape({ percent: number }), memory: shape({ total_bytes: number }), gpu: shape({ device: string }), disk: shape({ total_bytes: number }) });
const chatSummary = shape<ChatSummary>({ id: string, title: string, pinned: boolean, updated_at: string });
const chatGuard = shape<Chat>({ id: string, title: string, messages: arrayOf(shape<ChatMessage>({ id: string, role: string, content: string, variants: arrayOf(anything), reasoning_trace: arrayOf(anything) })) });
const rows = listOf<Metric>(shape({ step: number }));

export const api = {
  login: (identifier: string, password: string) => http<User>("/auth/login", body("POST", { identifier, password }), me, { session: false }),
  register: (payload: RegisterPayload) => http<User>("/auth/register", body("POST", payload), me, { session: false }),
  me: (signal?: AbortSignal) => http<User>("/auth/me", { signal }, me, { session: false, verifyVersion: true }),
  updateProfile: (payload: ProfilePayload) => http<User>("/auth/profile", body("PATCH", payload), me),
  sessions: () => http<UserSession[]>("/auth/sessions", {}, listOf<UserSession>(shape({ id: number, current: boolean }))),
  revokeSession: (id: number) => http<void>(`/auth/sessions/${id}`, body("DELETE")),
  revokeOtherSessions: () => http<{ revoked: number }>("/auth/sessions/revoke-others", body("POST")),
  users: (query: { query: string; role: Role | null; active: boolean | null; limit: number; offset: number }) =>
    http<UserPage>(`/users?${new URLSearchParams(Object.entries({ query: query.query, role: query.role ?? "", active: query.active === null ? "" : String(query.active), limit: String(query.limit), offset: String(query.offset) }).filter(([, value]) => value !== ""))}`, {}, userPage),
  updateUser: (id: string, changes: UserChanges) => http<User>(`/users/${id}`, body("PATCH", changes), me),
  resetUserPassword: (id: string) => http<{ temporary_password: string }>(`/users/${id}/reset-password`, body("POST"), shape({ temporary_password: string })),
  revokeUserSessions: (id: string) => http<{ revoked: number }>(`/users/${id}/revoke-sessions`, body("POST")),
  notifications: (query: NotificationQuery) =>
    http<NotificationPage>(`/notifications?${new URLSearchParams(Object.entries({ unread: String(query.unread), type: query.type ?? "", limit: String(query.limit), offset: String(query.offset) }).filter(([, value]) => value !== ""))}`, {}, notificationPage),
  unreadCount: () => http<{ unread: number }>("/notifications/unread-count", {}, shape({ unread: number })),
  readNotification: (id: string) => http<AppNotification>(`/notifications/${id}/read`, body("POST"), notification),
  readAllNotifications: () => http<{ updated: number }>("/notifications/read-all", body("POST")),
  deleteNotification: (id: string) => http<void>(`/notifications/${id}`, body("DELETE")),
  chatModel: () => http<ChatModelInfo | null>("/chat/model"),
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
  catalogRuns: (query: RunQuery = {}) =>
    http<RunPage>(`/catalog/runs?${new URLSearchParams(Object.entries({ ...query, limit: query.limit ?? 50, offset: query.offset ?? 0 }).filter(([, value]) => value !== undefined && value !== "").map(([key, value]) => [key, String(value)]))}`, {}, shape<RunPage>({ items: arrayOf(runEntry), total: number })),
  modelCatalog: () => http<CatalogModel[]>("/models/catalog", {}, listOf<CatalogModel>(catalogModel)),
  setAllowedModels: (ids: string[]) => http<CatalogModel[]>("/models/allowed", body("PUT", { ids }), listOf<CatalogModel>(catalogModel)),
  inferenceProfiles: () => http<InferenceProfile[]>("/inference/profiles", {}, listOf<InferenceProfile>(shape({ id: string, available: boolean, allowed: boolean }))),
  setAllowedProfiles: (ids: string[]) => http<InferenceProfile[]>("/inference/profiles/allowed", body("PUT", { ids }), listOf<InferenceProfile>(shape({ id: string, available: boolean, allowed: boolean }))),
  userModels: () => http<UserModel[]>("/chat/models", {}, listOf<UserModel>(userModel)),
  processEvents: (id: string) => http<{ level: string; message: string; created_at: string }[]>(`/processes/${encodeURIComponent(id)}/events`, {}, listOf(shape({ message: string, created_at: string }))),
  evaluationMetrics: () => http<MetricDefinition[]>("/evaluation/metrics", {}, listOf<MetricDefinition>(shape({ key: string, section: string, direction: string }))),
  evaluationGroups: (query: { q?: string; limit?: number; offset?: number } = {}) =>
    http<{ items: EvaluationGroupSummary[]; total: number }>(`/evaluation/groups?${new URLSearchParams(Object.entries({ ...query, limit: query.limit ?? 30, offset: query.offset ?? 0 }).filter(([, value]) => value !== undefined && value !== "").map(([key, value]) => [key, String(value)]))}`, {}, shape({ items: arrayOf(shape({ id: string, experiment: string })), total: number })),
  evaluationGroup: (id: string) => http<EvaluationGroup>(`/evaluation/groups/${encodeURIComponent(id)}`, {}, shape<EvaluationGroup>({ id: string, rows: arrayOf(shape({ results: record(number) })) })),
  compareEvaluations: (group: string, checkpoints: string[]) =>
    http<Comparison>(`/evaluation/compare?${new URLSearchParams([["group", group], ...checkpoints.map((item): [string, string] => ["checkpoint", item])])}`, {}, shape<Comparison>({ metrics: arrayOf(shape({ key: string })), verdict: shape({ status: string }) })),
  pipelines: () => http<Pipeline[]>("/pipelines", {}, listOf(pipelineGuard)),
  evaluations: () => http<Evaluation[]>("/evaluation", {}, listOf<Evaluation>(shape({ path: string, experiment: string, dataset: nullable(string), stage: nullable(string), model: nullable(string), checkpoint: nullable(string), results: record(number) }))),
  logs: (category?: string) => http<{ items: { category: string; source: string; message: string }[]; total: number }>(`/logs${category ? `?category=${category}` : ""}`, {}, shape({ items: arrayOf(shape({ message: string })), total: number })),
  chats: (query = "", archived = false) => http<ChatSummary[]>(`/chats?${new URLSearchParams({ query, archived: String(archived) })}`, {}, listOf<ChatSummary>(chatSummary)),
  chat: (id: string, limit = 100) => http<Chat>(`/chats/${id}?limit=${limit}`, {}, chatGuard),
  createChat: (payload?: { title?: string; model_name?: string; checkpoint?: string; system_prompt?: string; generation_settings?: GenerationSettings }) => http<ChatSummary>("/chats", body("POST", payload ?? {}), chatSummary),
  updateChat: (id: string, payload: { title?: string; model_name?: string; checkpoint?: string; system_prompt?: string; generation_settings?: GenerationSettings; pinned?: boolean; archived?: boolean }) => http<ChatSummary>(`/chats/${id}`, body("PATCH", payload), chatSummary),
  deleteChat: (id: string) => http<void>(`/chats/${id}`, body("DELETE")),
  selectVariant: (chatId: string, messageId: string, index: number) => http<ChatMessage>(`/chats/${chatId}/messages/${messageId}/variants/${index}`, body("POST")),
  exportChatUrl: (id: string, format: "md" | "json") => `/api/chats/${id}/export?format=${format}`,
};
