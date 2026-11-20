export function splitBlocks(text: string): string[] {
  const blocks: string[] = [];
  let current: string[] = [];
  let fenced = false;
  for (const line of text.split("\n")) {
    if (line.trimStart().startsWith("```")) fenced = !fenced;
    if (line === "" && !fenced && current.length > 0) {
      blocks.push(current.join("\n"));
      current = [];
    } else {
      current.push(line);
    }
  }
  if (current.length > 0) blocks.push(current.join("\n"));
  return blocks;
}
