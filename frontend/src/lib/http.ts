import { csrfToken } from "@/lib/utils";
import { isRecord, type Guard } from "@/lib/schema";

export const API_BASE = "/api";
const API_VERSION = "1";
const REQUEST_TIMEOUT_MS = 12_000;

export type ApiErrorCode =
  | "network"
  | "offline"
  | "incompatible"
  | "unavailable"
  | "html_response"
  | "invalid_response"
  | "unauthorized"
  | "forbidden"
  | "not_found"
  | "conflict"
  | "validation"
  | "rate_limited"
  | "server"
  | "http";

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly code: ApiErrorCode,
    public readonly detail: string | null = null,
  ) {
    super(detail ?? code);
  }
}

const STATUS_CODES: Record<number, ApiErrorCode> = {
  400: "validation",
  401: "unauthorized",
  403: "forbidden",
  404: "not_found",
  409: "conflict",
  422: "validation",
  429: "rate_limited",
  502: "unavailable",
  503: "unavailable",
  504: "unavailable",
};

type UnauthorizedHandler = () => void;
let onUnauthorized: UnauthorizedHandler = () => undefined;

export function setUnauthorizedHandler(handler: UnauthorizedHandler) {
  onUnauthorized = handler;
}

function codeFor(status: number): ApiErrorCode {
  return STATUS_CODES[status] ?? (status >= 500 ? "server" : "http");
}

function detailOf(payload: unknown): string | null {
  if (!isRecord(payload) || !("detail" in payload)) return null;
  const { detail } = payload;
  if (typeof detail === "string") return detail;
  return Array.isArray(detail) ? "invalid_request" : null;
}

async function failure(response: Response): Promise<ApiError> {
  const type = response.headers.get("content-type") ?? "";
  if (type.includes("application/json")) {
    const payload: unknown = await response.json().catch(() => null);
    return new ApiError(response.status, codeFor(response.status), detailOf(payload));
  }
  const text = await response.text().catch(() => "");
  const code = /^\s*</.test(text) ? "html_response" : codeFor(response.status);
  return new ApiError(response.status, code, null);
}

export async function http<T>(path: string, init: RequestInit = {}, guard?: Guard<T>, options: { session?: boolean; verifyVersion?: boolean } = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body) headers.set("Content-Type", "application/json");
  headers.set("Accept", "application/json");
  if (init.method && !["GET", "HEAD"].includes(init.method)) {
    const token = csrfToken();
    if (token) headers.set("X-CSRF-Token", token);
  }
  let response: Response;
  try {
    const timeout = AbortSignal.timeout(REQUEST_TIMEOUT_MS);
    const signal = init.signal ? AbortSignal.any([init.signal, timeout]) : timeout;
    response = await fetch(`${API_BASE}${path}`, { ...init, headers, signal, redirect: "error", credentials: "same-origin" });
  } catch {
    throw new ApiError(0, "network");
  }
  if (options.verifyVersion && response.status < 500 && response.headers.get("X-Studio-API-Version") !== API_VERSION) {
    throw new ApiError(response.status, "incompatible");
  }
  if (!response.ok) {
    const error = await failure(response);
    if (error.code === "unauthorized" && options.session !== false) onUnauthorized();
    throw error;
  }
  if (response.status === 204) return undefined as T;
  const type = response.headers.get("content-type") ?? "";
  if (!type.includes("application/json")) {
    const text = await response.text().catch(() => "");
    throw new ApiError(response.status, /^\s*</.test(text) ? "html_response" : "invalid_response");
  }
  const payload: unknown = await response.json().catch(() => {
    throw new ApiError(response.status, "invalid_response");
  });
  if (guard && !guard(payload)) throw new ApiError(response.status, "invalid_response");
  return payload as T;
}

export const body = (method: string, payload?: unknown): RequestInit => ({
  method,
  body: payload === undefined ? undefined : JSON.stringify(payload),
});
