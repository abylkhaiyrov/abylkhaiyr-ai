export type Guard<T> = (value: unknown) => value is T;
type Check = (value: unknown) => boolean;

export const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

export const string: Check = (value) => typeof value === "string";
export const number: Check = (value) => typeof value === "number";
export const boolean: Check = (value) => typeof value === "boolean";
export const anything: Check = () => true;
export const nullable = (check: Check): Check => (value) => value === null || check(value);
export const optional = (check: Check): Check => (value) => value === undefined || check(value);
export const arrayOf = (check: Check): Check => (value) => Array.isArray(value) && value.every(check);
export const record = (check: Check): Check => (value) => isRecord(value) && Object.values(value).every(check);

export function shape<T>(fields: Record<string, Check>): Guard<T> {
  return (value): value is T => isRecord(value) && Object.entries(fields).every(([key, check]) => check(value[key]));
}

export const listOf = <T>(item: Check): Guard<T[]> => ((value: unknown): value is T[] => arrayOf(item)(value));
