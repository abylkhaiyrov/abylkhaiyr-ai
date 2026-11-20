import { describe, expect, it } from "vitest";
import { validateEmail, validateName, validatePassword, validateUsername } from "@/lib/validation";

describe("validation", () => {
  it("accepts normalized emails and rejects malformed ones", () => {
    expect(validateEmail(" Alice@Example.COM ")).toBeNull();
    expect(validateEmail("alice@example")).toBe("invalidEmail");
    expect(validateEmail("")).toBe("required");
  });

  it("enforces username rules after normalization", () => {
    expect(validateUsername("Alice_01")).toBeNull();
    expect(validateUsername("a")).toBe("invalidUsername");
    expect(validateUsername("bad name")).toBe("invalidUsername");
  });

  it("requires long passwords that differ from the username", () => {
    expect(validatePassword("short")).toBe("weakPassword");
    expect(validatePassword("alice-alice-alice", "ALICE-ALICE-ALICE")).toBe("weakPassword");
    expect(validatePassword("a long enough password", "alice")).toBeNull();
  });

  it("limits names", () => {
    expect(validateName("   ")).toBe("required");
    expect(validateName("x".repeat(65))).toBe("nameTooLong");
  });
});
