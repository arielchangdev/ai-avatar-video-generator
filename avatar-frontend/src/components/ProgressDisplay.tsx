import type { ErrorMessage, TaskProgress } from "../types";

interface Props {
  progress: TaskProgress | null;
  error: ErrorMessage | null;
  onRetry: (fromStage?: string) => void;
}

// Shows the current stage name and percentage bar. On failure it renders the
// failed stage, error category, and a retry button.
export function ProgressDisplay({ progress, error, onRetry }: Props) {
  const failed = progress?.state === "failed" || error != null;

  if (failed) {
    const stage = error?.stage ?? progress?.error_stage ?? undefined;
    const category = error?.error_category ?? "PipelineError";
    const message = error?.message ?? progress?.error_message ?? "生成失敗";
    return (
      <div className="progress-display error" role="alert">
        <h3>生成失敗</h3>
        <p>失敗階段：{stage ?? "未知"}</p>
        <p>錯誤類別：{category}</p>
        <p>{message}</p>
        <button type="button" onClick={() => onRetry(stage)}>
          從失敗階段重試
        </button>
      </div>
    );
  }

  const pct = progress?.percentage ?? 0;
  const stageName = progress?.current_stage ?? "準備中";

  return (
    <div className="progress-display">
      <h3>生成進度</h3>
      <p className="stage-name">目前階段：{stageName}</p>
      <div className="progress-bar-track">
        <div
          className="progress-bar-fill"
          style={{ width: `${pct}%` }}
          role="progressbar"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>
      <p className="percentage">{pct}%</p>
    </div>
  );
}
