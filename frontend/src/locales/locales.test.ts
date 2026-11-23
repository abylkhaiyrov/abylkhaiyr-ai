import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { en } from "@/locales/en";
import { ru } from "@/locales/ru";

type Tree = { [key: string]: string | Tree };

function leaves(tree: Tree, prefix = ""): string[] {
  return Object.entries(tree).flatMap(([key, value]) => (typeof value === "string" ? [`${prefix}${key}`] : leaves(value, `${prefix}${key}.`)));
}

function sources(directory: string): string[] {
  return readdirSync(directory).flatMap((name) => {
    const path = join(directory, name);
    if (statSync(path).isDirectory()) return name === "locales" ? [] : sources(path);
    return /\.tsx?$/.test(name) && !name.endsWith(".test.ts") ? [path] : [];
  });
}

describe("locales", () => {
  const russian = new Set(leaves(ru));
  const english = new Set(leaves(en));

  it("have identical keys", () => {
    expect([...russian].filter((key) => !english.has(key))).toEqual([]);
    expect([...english].filter((key) => !russian.has(key))).toEqual([]);
  });

  it("cover every statically used key", () => {
    const used = new Set<string>();
    for (const file of sources(join(__dirname, ".."))) {
      const text = readFileSync(file, "utf8");
      for (const match of text.matchAll(/\bt\(\s*"([A-Za-z0-9_.]+)"/g)) used.add(match[1]);
      for (const match of text.matchAll(/(?:titleKey|unitKey):\s*"([A-Za-z0-9_.]+)"/g)) used.add(match[1]);
    }
    expect([...used].filter((key) => !russian.has(key))).toEqual([]);
  });

  it("leave no Latin-only values in Russian outside technical terms", () => {
    const allowed = /^(Abylkhaiyr AI Studio|SFT|DPO|GSM8K|Pretrain|Production|Pipeline|CPU|Swap|Top-P|Top-K|Temperature|Checkpoint|Perplexity|Learning rate|Train loss|Validation loss|Test loss|Email|Test perplexity|Training loss|Lite|Medium|High|Very High|Reasoning Engine|—)$/;
    const offenders = leaves(ru).filter((key) => {
      const value = key.split(".").reduce<unknown>((node, part) => (node as Tree)[part], ru) as string;
      return /^[\x20-\x7E]+$/.test(value) && !allowed.test(value) && !/\{\{/.test(value);
    });
    expect(offenders).toEqual([]);
  });
});
