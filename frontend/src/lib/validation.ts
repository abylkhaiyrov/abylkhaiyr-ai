export type FieldCode =
  | "required"
  | "invalidEmail"
  | "invalidUsername"
  | "weakPassword"
  | "passwordMismatch"
  | "nameTooLong";

const EMAIL = /^[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}$/;
const USERNAME = /^[a-z0-9][a-z0-9_.-]{2,31}$/;
export const PASSWORD_MIN = 12;
const NAME_MAX = 64;

export const normalizeUsername = (value: string) => value.trim().toLowerCase();
export const normalizeEmail = (value: string) => value.trim().toLowerCase();

export function validateName(value: string): FieldCode | null {
  const name = value.trim();
  if (!name) return "required";
  return name.length > NAME_MAX ? "nameTooLong" : null;
}

export function validateEmail(value: string): FieldCode | null {
  const email = normalizeEmail(value);
  if (!email) return "required";
  return email.length <= 254 && EMAIL.test(email) ? null : "invalidEmail";
}

export function validateUsername(value: string): FieldCode | null {
  const username = normalizeUsername(value);
  if (!username) return "required";
  return USERNAME.test(username) ? null : "invalidUsername";
}

export function validatePassword(value: string, username = ""): FieldCode | null {
  if (!value) return "required";
  if (value.length < PASSWORD_MIN || value.toLowerCase() === username.trim().toLowerCase()) return "weakPassword";
  return null;
}

export function firstError<T extends string>(errors: Record<T, FieldCode | null>): boolean {
  return Object.values(errors).some(Boolean);
}
