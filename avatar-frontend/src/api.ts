// Thin API client for the FastAPI backend. Uses relative URLs so the Vite
// dev-server proxy (or same-origin production deploy) routes to the backend.

import type { CreateTaskResponse, TaskProgress } from "./types";

const BASE = "/api/v1/tasks";

export interface CreateTaskParams {
  voiceSample: File;
  appearanceAsset: File;
  scriptText: string;
  retryFromStage?: string;
}

export async function createTask(
  params: CreateTaskParams,
): Promise<CreateTaskResponse> {
  const form = new FormData();
  form.append("voice_sample", params.voiceSample);
  form.append("appearance_asset", params.appearanceAsset);
  form.append("script_text", params.scriptText);
  if (params.retryFromStage) {
    form.append("retry_from_stage", params.retryFromStage);
  }

  const resp = await fetch(BASE, { method: "POST", body: form });
  if (!resp.ok) {
    const detail = await resp.json().catch(() => null);
    const errors: string[] =
      detail && detail.detail && detail.detail.errors
        ? detail.detail.errors
        : [detail?.detail ?? `Request failed (${resp.status})`];
    throw new SubmissionError(errors);
  }
  return resp.json();
}

export async function getTask(taskId: string): Promise<TaskProgress> {
  const resp = await fetch(`${BASE}/${taskId}`);
  if (!resp.ok) {
    throw new Error(`Failed to fetch task (${resp.status})`);
  }
  return resp.json();
}

export async function deleteTask(taskId: string): Promise<void> {
  const resp = await fetch(`${BASE}/${taskId}`, { method: "DELETE" });
  if (!resp.ok && resp.status !== 404) {
    throw new Error(`Failed to delete task (${resp.status})`);
  }
}

export function videoDownloadUrl(taskId: string): string {
  return `${BASE}/${taskId}/download/video`;
}

export function audioDownloadUrl(taskId: string): string {
  return `${BASE}/${taskId}/download/audio`;
}

export function progressSocketUrl(taskId: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/api/v1/ws/tasks/${taskId}/progress`;
}

// Carries the list of validation errors returned by the backend so the UI can
// render each failed item.
export class SubmissionError extends Error {
  errors: string[];
  constructor(errors: string[]) {
    super(errors.join("; "));
    this.name = "SubmissionError";
    this.errors = errors;
  }
}
