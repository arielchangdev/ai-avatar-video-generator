import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ProgressDisplay } from "./ProgressDisplay";
import type { TaskProgress } from "../types";

function progress(overrides: Partial<TaskProgress>): TaskProgress {
  return {
    task_id: "t1",
    state: "processing",
    current_stage: "語音合成",
    percentage: 40,
    error_message: null,
    error_stage: null,
    ...overrides,
  };
}

describe("ProgressDisplay", () => {
  it("shows the current stage and percentage", () => {
    render(<ProgressDisplay progress={progress({})} error={null} onRetry={vi.fn()} />);
    expect(screen.getByText(/語音合成/)).toBeInTheDocument();
    const bar = screen.getByRole("progressbar");
    expect(bar).toHaveAttribute("aria-valuenow", "40");
  });

  it("renders a failure state with retry button", () => {
    const onRetry = vi.fn();
    render(
      <ProgressDisplay
        progress={progress({
          state: "failed",
          error_stage: "唇形同步",
          error_message: "SadTalker failed",
        })}
        error={null}
        onRetry={onRetry}
      />,
    );
    expect(screen.getByText(/失敗階段：唇形同步/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /從失敗階段重試/ }));
    expect(onRetry).toHaveBeenCalledWith("唇形同步");
  });

  it("prefers a WebSocket error notification when present", () => {
    render(
      <ProgressDisplay
        progress={progress({})}
        error={{
          type: "error",
          stage: "影片合成",
          error_category: "VideoComposeError",
          message: "FFmpeg mux failed",
          recoverable: true,
        }}
        onRetry={vi.fn()}
      />,
    );
    expect(screen.getByText(/錯誤類別：VideoComposeError/)).toBeInTheDocument();
    expect(screen.getByText(/FFmpeg mux failed/)).toBeInTheDocument();
  });

  it("defaults to preparing state before any progress arrives", () => {
    render(<ProgressDisplay progress={null} error={null} onRetry={vi.fn()} />);
    expect(screen.getByText(/準備中/)).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "0");
  });
});
