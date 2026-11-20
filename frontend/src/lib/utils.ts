import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function csrfToken() {
  return document.cookie
    .split("; ")
    .find((item) => item.startsWith("abylkhaiyr_csrf="))
    ?.split("=")[1];
}

export function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}
