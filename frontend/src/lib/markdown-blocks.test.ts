import { describe, expect, it } from "vitest";
import { splitBlocks } from "@/lib/markdown-blocks";

describe("splitBlocks", () => {
  it("splits paragraphs but keeps fenced code intact", () => {
    const text = "one\n\ntwo\n\n```py\na = 1\n\nb = 2\n```\n\nthree";
    expect(splitBlocks(text)).toEqual(["one", "two", "```py\na = 1\n\nb = 2\n```", "three"]);
  });

  it("keeps an unfinished fence as one growing block", () => {
    expect(splitBlocks("intro\n\n```js\nlet x\n\nlet y")).toEqual(["intro", "```js\nlet x\n\nlet y"]);
  });
});
