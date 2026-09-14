import { describe, it, expect } from "vitest";
import {
  appearanceKind,
  validateAppearanceFile,
  validateScript,
  validateVoiceFile,
} from "./validation";

function fakeFile(name: string, sizeBytes: number): File {
  const blob = new Blob([new Uint8Array(Math.min(sizeBytes, 1024))]);
  const file = new File([blob], name);
  // Override size (constructing a real large file is wasteful).
  Object.defineProperty(file, "size", { value: sizeBytes });
  return file;
}

describe("validateVoiceFile", () => {
  it("accepts a valid wav within size limit", () => {
    const res = validateVoiceFile(fakeFile("v.wav", 1024));
    expect(res.valid).toBe(true);
    expect(res.errors).toHaveLength(0);
  });

  it("rejects an unsupported format", () => {
    const res = validateVoiceFile(fakeFile("v.ogg", 1024));
    expect(res.valid).toBe(false);
    expect(res.errors.join()).toContain("格式");
  });

  it("rejects files over 50MB", () => {
    const res = validateVoiceFile(fakeFile("v.wav", 51 * 1024 * 1024));
    expect(res.valid).toBe(false);
    expect(res.errors.join()).toContain("50MB");
  });

  it("rejects an empty file", () => {
    const res = validateVoiceFile(fakeFile("v.wav", 0));
    expect(res.valid).toBe(false);
  });
});

describe("validateAppearanceFile", () => {
  it("accepts a valid png image", () => {
    expect(validateAppearanceFile(fakeFile("f.png", 1024)).valid).toBe(true);
  });

  it("accepts a valid mp4 video", () => {
    expect(validateAppearanceFile(fakeFile("f.mp4", 1024)).valid).toBe(true);
  });

  it("rejects an image over 20MB", () => {
    const res = validateAppearanceFile(fakeFile("f.png", 21 * 1024 * 1024));
    expect(res.valid).toBe(false);
    expect(res.errors.join()).toContain("20MB");
  });

  it("rejects a video over 200MB", () => {
    const res = validateAppearanceFile(fakeFile("f.mp4", 201 * 1024 * 1024));
    expect(res.valid).toBe(false);
    expect(res.errors.join()).toContain("200MB");
  });

  it("rejects unsupported formats", () => {
    expect(validateAppearanceFile(fakeFile("f.gif", 1024)).valid).toBe(false);
  });
});

describe("validateScript", () => {
  it("accepts normal text", () => {
    expect(validateScript("你好世界").valid).toBe(true);
  });

  it("rejects empty text", () => {
    expect(validateScript("").valid).toBe(false);
  });

  it("rejects whitespace-only text", () => {
    expect(validateScript("   \t ").valid).toBe(false);
  });

  it("rejects text over 10000 chars", () => {
    const res = validateScript("a".repeat(10001));
    expect(res.valid).toBe(false);
    expect(res.errors.join()).toContain("10000");
  });

  it("accepts exactly 10000 chars", () => {
    expect(validateScript("a".repeat(10000)).valid).toBe(true);
  });
});

describe("appearanceKind", () => {
  it("classifies image and video by extension", () => {
    expect(appearanceKind(fakeFile("f.png", 1))).toBe("image");
    expect(appearanceKind(fakeFile("f.mov", 1))).toBe("video");
    expect(appearanceKind(fakeFile("f.txt", 1))).toBe("unknown");
  });
});
