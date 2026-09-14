import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { SubmissionForm } from "./SubmissionForm";

function fakeFile(name: string, sizeBytes = 1024): File {
  const file = new File([new Uint8Array(16)], name);
  Object.defineProperty(file, "size", { value: sizeBytes });
  return file;
}

function uploadTo(labelText: string, file: File) {
  const input = screen.getByLabelText(labelText) as HTMLInputElement;
  fireEvent.change(input, { target: { files: [file] } });
}

describe("SubmissionForm", () => {
  it("disables confirm until all inputs are valid", () => {
    render(<SubmissionForm onSubmit={vi.fn()} submitting={false} serverErrors={[]} />);
    const button = screen.getByRole("button", { name: /確認並開始生成/ });
    expect(button).toBeDisabled();
  });

  it("shows the material summary once all inputs are valid", () => {
    render(<SubmissionForm onSubmit={vi.fn()} submitting={false} serverErrors={[]} />);

    uploadTo("聲音樣本", fakeFile("voice.wav"));
    uploadTo("人物外觀（圖片或影片）", fakeFile("face.png"));
    fireEvent.change(screen.getByLabelText("講稿內容"), {
      target: { value: "測試講稿內容" },
    });

    const summary = screen.getByTestId("material-summary");
    expect(summary).toHaveTextContent("圖片");
    expect(summary).toHaveTextContent("講稿字數：6 字");
    expect(screen.getByRole("button", { name: /確認並開始生成/ })).toBeEnabled();
  });

  it("calls onSubmit with the combined payload on confirm", () => {
    const onSubmit = vi.fn();
    render(<SubmissionForm onSubmit={onSubmit} submitting={false} serverErrors={[]} />);

    uploadTo("聲音樣本", fakeFile("voice.mp3"));
    uploadTo("人物外觀（圖片或影片）", fakeFile("clip.mp4"));
    fireEvent.change(screen.getByLabelText("講稿內容"), {
      target: { value: "你好" },
    });

    fireEvent.click(screen.getByRole("button", { name: /確認並開始生成/ }));
    expect(onSubmit).toHaveBeenCalledTimes(1);
    const payload = onSubmit.mock.calls[0][0];
    expect(payload.voiceSample.name).toBe("voice.mp3");
    expect(payload.appearanceAsset.name).toBe("clip.mp4");
    expect(payload.scriptText).toBe("你好");
  });

  it("renders server-side validation errors", () => {
    render(
      <SubmissionForm
        onSubmit={vi.fn()}
        submitting={false}
        serverErrors={["Missing: voice_sample"]}
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("Missing: voice_sample");
  });

  it("blocks submission when the voice file is invalid", () => {
    render(<SubmissionForm onSubmit={vi.fn()} submitting={false} serverErrors={[]} />);
    uploadTo("聲音樣本", fakeFile("voice.ogg"));
    uploadTo("人物外觀（圖片或影片）", fakeFile("face.png"));
    fireEvent.change(screen.getByLabelText("講稿內容"), {
      target: { value: "測試" },
    });
    expect(screen.getByRole("button", { name: /確認並開始生成/ })).toBeDisabled();
  });
});
