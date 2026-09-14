// Client-side validation mirroring the backend validators. This gives users
// immediate feedback; the backend remains the source of truth on submit.

import {
  IMAGE_FORMATS,
  IMAGE_MAX_MB,
  SCRIPT_MAX_CHARS,
  VIDEO_FORMATS,
  VIDEO_MAX_MB,
  VOICE_FORMATS,
  VOICE_MAX_MB,
} from "./types";

export interface FieldValidation {
  valid: boolean;
  errors: string[];
}

function extension(filename: string): string {
  const idx = filename.lastIndexOf(".");
  return idx >= 0 ? filename.slice(idx + 1).toLowerCase() : "";
}

const MB = 1024 * 1024;

export function validateVoiceFile(file: File): FieldValidation {
  const errors: string[] = [];
  const ext = extension(file.name);
  if (!VOICE_FORMATS.includes(ext)) {
    errors.push(`不支援的音訊格式：.${ext || "未知"}（支援 ${VOICE_FORMATS.join("、")}）`);
  }
  if (file.size > VOICE_MAX_MB * MB) {
    errors.push(`音訊檔案超過 ${VOICE_MAX_MB}MB 上限`);
  }
  if (file.size === 0) {
    errors.push("音訊檔案為空");
  }
  return { valid: errors.length === 0, errors };
}

export function validateAppearanceFile(file: File): FieldValidation {
  const errors: string[] = [];
  const ext = extension(file.name);
  const isImage = IMAGE_FORMATS.includes(ext);
  const isVideo = VIDEO_FORMATS.includes(ext);

  if (!isImage && !isVideo) {
    errors.push(
      `不支援的外觀素材格式：.${ext || "未知"}（圖片 ${IMAGE_FORMATS.join("、")}；影片 ${VIDEO_FORMATS.join("、")}）`,
    );
  } else if (isImage && file.size > IMAGE_MAX_MB * MB) {
    errors.push(`圖片檔案超過 ${IMAGE_MAX_MB}MB 上限`);
  } else if (isVideo && file.size > VIDEO_MAX_MB * MB) {
    errors.push(`影片檔案超過 ${VIDEO_MAX_MB}MB 上限`);
  }
  if (file.size === 0) {
    errors.push("外觀素材檔案為空");
  }
  return { valid: errors.length === 0, errors };
}

export function validateScript(text: string): FieldValidation {
  const errors: string[] = [];
  if (text.length < 1) {
    errors.push("講稿不得為空");
  }
  if (text.length > SCRIPT_MAX_CHARS) {
    errors.push(`講稿字數超過 ${SCRIPT_MAX_CHARS} 上限（目前 ${text.length} 字）`);
  }
  if (text.length >= 1 && text.trim() === "") {
    errors.push("講稿不得僅含空白字元");
  }
  return { valid: errors.length === 0, errors };
}

export function appearanceKind(file: File): "image" | "video" | "unknown" {
  const ext = extension(file.name);
  if (IMAGE_FORMATS.includes(ext)) return "image";
  if (VIDEO_FORMATS.includes(ext)) return "video";
  return "unknown";
}
