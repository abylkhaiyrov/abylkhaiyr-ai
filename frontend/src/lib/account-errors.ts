import { ApiError } from "@/lib/http";

const FIELD_BY_DETAIL: Record<string, string> = {
  email_taken: "email",
  invalid_email: "email",
  username_taken: "username",
  invalid_username: "username",
  weak_password: "password",
  invalid_first_name: "firstName",
  invalid_last_name: "lastName",
  invalid_password: "currentPassword",
};

export function fieldOf(error: unknown): string | null {
  return error instanceof ApiError && error.detail ? (FIELD_BY_DETAIL[error.detail] ?? null) : null;
}
