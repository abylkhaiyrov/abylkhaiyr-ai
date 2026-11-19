import { csrfToken } from "@/lib/utils";

export type SystemSnapshot = {
  recorded_at: string;
  cpu: { percent: number; per_core_percent: number[]; logical_cores: number | null; frequency_mhz: number | null };
  memory: { total_bytes: number; used_bytes: number; available_bytes: number; python_process_bytes: number; swap_total_bytes: number; swap_used_bytes: number; pressure: number | null };
  gpu: { device: string; allocated_bytes: number | null; driver_allocated_bytes: number | null; peak_bytes: number | null; supported_dtypes: string[] };
  disk: { total_bytes: number; used_bytes: number; free_bytes: number; datasets_bytes: number; checkpoints_bytes: number; database_bytes: number; logs_bytes: number };
};

export type Dashboard = {
  system_status: string;
  active_model: Model | null;
  inference_model: string | null;
  training: {
    stage: string | null;
    process_id: string | null;
    step: number | null;
    max_steps: number | null;
    tokens_seen: number | null;
    train_loss: number | null;
    validation_loss: number | null;
    perplexity: number | null;
    throughput: number | null;
    eta_seconds: number | null;
    started_at: string | null;
    model_name: string | null;
    checkpoint: string | null;
  };
  active_process: Process | null;
  pipeline: { id: string; status: string }[];
  system: SystemSnapshot;
  counts: { active_processes: number; checkpoints: number };
  events: Event[];
};

export type Metric = Record<string, number>;

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
};

export type Model = {
  id: string;
  name: string;
  path: string;
  step: number | null;
  tokens_seen: number | null;
  created_at: string;
  weights_bytes: number | null;
  parameters: number | null;
  vocab_size: number | null;
  context_length: number | null;
  architecture: Record<string, unknown>;
  metrics: Metric;
};

export type Event = { process_id: string; level: string; message: string; created_at: string };
export type Dataset = { name: string; path: string; size_bytes: number; manifest: Record<string, unknown> };
export type Experiment = { name: string; results: Metric; [key: string]: unknown };
export type Evaluation = { path: string; results: Metric };
export type ChatSummary = { id: string; title: string; model_name: string | null; checkpoint: string | null; system_prompt: string; generation_settings: Record<string, unknown>; created_at: string; updated_at: string };
export type ChatMessage = { id: string; role: "system" | "user" | "assistant" | "tool"; content: string; input_tokens: number | null; output_tokens: number | null; generation_ms: number | null; tokens_per_second: number | null; created_at: string };
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

export const api = {
  login: (username: string, password: string) => request<{ username: string; requires_password_change: boolean }>("/auth/login", { method: "POST", body: JSON.stringify({ username, password }) }),
  me: () => request<{ username: string; requires_password_change: boolean }>("/auth/me"),
  logout: () => request<void>("/auth/logout", { method: "POST" }),
  changePassword: (currentPassword: string, newPassword: string) => request<void>("/auth/password", { method: "PUT", body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }) }),
  dashboard: () => request<Dashboard>("/dashboard"),
  system: () => request<SystemSnapshot>("/system"),
  systemHistory: (seconds: number) => request<SystemSnapshot[]>(`/system/history?seconds=${seconds}`),
  processes: () => request<Process[]>("/processes"),
  startTraining: (configPath: string, stage?: string) => request<Process>("/training", { method: "POST", body: JSON.stringify({ config_path: configPath, stage }) }),
  pauseTraining: (id: string) => request<Process>(`/training/${id}/pause`, { method: "POST" }),
  stopTraining: (id: string) => request<Process>(`/training/${id}/stop`, { method: "POST" }),
  metrics: () => request<Metric[]>("/training/metrics"),
  models: () => request<Model[]>("/models"),
  datasets: () => request<Dataset[]>("/datasets"),
  experiments: () => request<Experiment[]>("/experiments"),
  evaluations: () => request<Evaluation[]>("/evaluation"),
  logs: () => request<{ items: { category: string; source: string; message: string }[]; total: number }>("/logs"),
  chats: (query?: string) => request<ChatSummary[]>(`/chats${query ? `?query=${encodeURIComponent(query)}` : ""}`),
  chat: (id: string) => request<Chat>(`/chats/${id}`),
  createChat: () => request<ChatSummary>("/chats", { method: "POST", body: JSON.stringify({}) }),
  renameChat: (id: string, title: string) => request<ChatSummary>(`/chats/${id}`, { method: "PATCH", body: JSON.stringify({ title }) }),
  deleteChat: (id: string) => request<void>(`/chats/${id}`, { method: "DELETE" }),
  sendMessage: (id: string, content: string) => request<{ user: ChatMessage; assistant: ChatMessage }>(`/chats/${id}/messages`, { method: "POST", body: JSON.stringify({ content }) }),
};
