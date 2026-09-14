import { useState } from "react";
import { SubmissionForm, type SubmissionPayload } from "./components/SubmissionForm";
import { ProgressDisplay } from "./components/ProgressDisplay";
import { VideoResult } from "./components/VideoResult";
import { useProgress } from "./useProgress";
import { createTask, SubmissionError } from "./api";

// Application flow: submit materials -> track progress -> preview & download.
export function App() {
  const [taskId, setTaskId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [serverErrors, setServerErrors] = useState<string[]>([]);
  const [lastPayload, setLastPayload] = useState<SubmissionPayload | null>(null);

  const { progress, error } = useProgress(taskId);
  const completed = progress?.state === "completed";

  async function submit(payload: SubmissionPayload, retryFromStage?: string) {
    setSubmitting(true);
    setServerErrors([]);
    try {
      const resp = await createTask({
        voiceSample: payload.voiceSample,
        appearanceAsset: payload.appearanceAsset,
        scriptText: payload.scriptText,
        retryFromStage,
      });
      setLastPayload(payload);
      setTaskId(resp.task_id);
    } catch (e) {
      if (e instanceof SubmissionError) {
        setServerErrors(e.errors);
      } else {
        setServerErrors([(e as Error).message]);
      }
    } finally {
      setSubmitting(false);
    }
  }

  function handleRetry(fromStage?: string) {
    if (lastPayload) {
      setTaskId(null);
      submit(lastPayload, fromStage);
    }
  }

  return (
    <main className="app">
      <h1>AI 虛擬人物影片生成器</h1>
      <p className="subtitle">本地零費用生成 · 上傳聲音與外觀，輸入講稿即可</p>

      {!taskId && (
        <SubmissionForm
          onSubmit={(p) => submit(p)}
          submitting={submitting}
          serverErrors={serverErrors}
        />
      )}

      {taskId && !completed && (
        <ProgressDisplay progress={progress} error={error} onRetry={handleRetry} />
      )}

      {taskId && completed && <VideoResult taskId={taskId} />}

      {taskId && (
        <button
          type="button"
          className="new-task-button"
          onClick={() => {
            setTaskId(null);
            setServerErrors([]);
          }}
        >
          建立新任務
        </button>
      )}
    </main>
  );
}
